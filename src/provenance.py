"""Reverse-image search, bounded web crawling, and provenance extraction."""

import ipaddress
import json
import re
import socket
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.robotparser import RobotFileParser

from .llm import parse_json_output
from .prompt import PROVENANCE_SYSTEM, PROVENANCE_USER, render_prompt


def _validate_public_url(url: str) -> str:
    """Reject non-Web and non-public targets before crawling them."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")
    candidate = url.strip()
    if len(candidate) > 4096:
        raise ValueError("URL is too long")
    parsed = urlsplit(candidate)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http and https URLs may be crawled")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("URL must contain a public host and no credentials")
    hostname = parsed.hostname.casefold()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("Local hosts may not be crawled")
    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        addresses = socket.getaddrinfo(
            parsed.hostname, port, type=socket.SOCK_STREAM
        )
    except (OSError, ValueError) as exc:
        raise ValueError("URL host could not be resolved") from exc
    if not addresses:
        raise ValueError("URL host did not resolve to an address")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0].split("%", 1)[0])
        if not ip.is_global:
            raise ValueError("URL resolves to a non-public address")
    return urlunsplit(parsed)


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_public_url(urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _PageParser(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "template"}
    _DATE_KEYS = (
        "article:published_time",
        "date",
        "datepublished",
        "datecreated",
        "pubdate",
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.in_title = False
        self.in_caption = False
        self.title_parts: List[str] = []
        self.caption_parts: List[str] = []
        self.captions: List[str] = []
        self.text_parts: List[str] = []
        self.meta: Dict[str, str] = {}
        self.dates: List[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        tag = tag.casefold()
        values = {str(key).casefold(): str(value) for key, value in attrs if value}
        if tag in self._SKIP:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = True
        elif tag == "figcaption":
            self.in_caption = True
            self.caption_parts = []
        elif tag == "meta":
            key = (values.get("property") or values.get("name") or "").casefold()
            content = values.get("content", "").strip()
            if key and content and key not in self.meta:
                self.meta[key] = content
        elif tag == "time":
            value = values.get("datetime", "").strip()
            if value:
                self.dates.append(value)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in self._SKIP and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = False
        elif tag == "figcaption" and self.in_caption:
            caption = _compact_text(" ".join(self.caption_parts))
            if caption:
                self.captions.append(caption)
            self.in_caption = False
            self.caption_parts = []

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        value = data.strip()
        if not value:
            return
        self.text_parts.append(value)
        if self.in_title:
            self.title_parts.append(value)
        if self.in_caption:
            self.caption_parts.append(value)

    def result(self, max_text_chars: int) -> Dict[str, str]:
        title = self.meta.get("og:title") or " ".join(self.title_parts)
        caption = (
            self.meta.get("og:description")
            or self.meta.get("description")
            or (self.captions[0] if self.captions else "")
        )
        date = ""
        for key in self._DATE_KEYS:
            if self.meta.get(key):
                date = self.meta[key]
                break
        if not date and self.dates:
            date = self.dates[0]
        return {
            "title": _compact_text(title)[:500],
            "date": _compact_text(date)[:200],
            "caption": _compact_text(caption)[:2000],
            "text": _compact_text(" ".join(self.text_parts))[:max_text_chars],
        }


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


class WebCrawler:
    """Fetch one public page safely and extract compact provenance context."""

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        max_bytes: int = 2_000_000,
        max_text_chars: int = 4000,
        user_agent: str = "TRACE-Fact-Provenance/1.0",
        respect_robots: bool = True,
    ) -> None:
        self.timeout = float(timeout)
        self.max_bytes = int(max_bytes)
        self.max_text_chars = int(max_text_chars)
        self.user_agent = str(user_agent).strip()
        self.respect_robots = bool(respect_robots)
        if self.timeout <= 0 or self.max_bytes < 1 or self.max_text_chars < 1:
            raise ValueError("Crawler timeout and size limits must be positive")
        if not self.user_agent:
            raise ValueError("Crawler user_agent must not be empty")
        self.opener = build_opener(_SafeRedirectHandler())

    def crawl(self, url: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "url": str(url),
            "title": "",
            "date": "",
            "caption": "",
            "text": "",
            "errors": [],
        }
        try:
            safe_url = _validate_public_url(url)
            if self.respect_robots and not self._robots_allowed(safe_url):
                result["errors"].append("Crawl blocked by robots.txt")
                return result
            final_url, content_type, charset, payload = self._fetch(
                safe_url, self.max_bytes
            )
            result["url"] = final_url
            if content_type not in {"text/html", "application/xhtml+xml"}:
                result["errors"].append(
                    "Unsupported crawl content type: %s" % content_type
                )
                return result
            parser = _PageParser()
            parser.feed(payload.decode(charset or "utf-8", errors="replace"))
            result.update(parser.result(self.max_text_chars))
        except Exception as exc:
            result["errors"].append(
                "Web crawl failed: %s: %s" % (type(exc).__name__, exc)
            )
        return result

    def _robots_allowed(self, url: str) -> bool:
        parsed = urlsplit(url)
        robots_url = urlunsplit((parsed.scheme, parsed.netloc, "/robots.txt", "", ""))
        try:
            _, content_type, charset, payload = self._fetch(robots_url, 256_000)
            if content_type not in {"text/plain", "text/html", ""}:
                return True
            robots = RobotFileParser()
            robots.set_url(robots_url)
            robots.parse(
                payload.decode(charset or "utf-8", errors="replace").splitlines()
            )
            return robots.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def _fetch(self, url: str, max_bytes: int):
        request = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.5",
            },
        )
        with self.opener.open(request, timeout=self.timeout) as response:
            final_url = _validate_public_url(response.geturl())
            content_type = response.headers.get_content_type().casefold()
            charset = response.headers.get_content_charset() or "utf-8"
            payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError("Crawl response exceeded the configured byte limit")
        return final_url, content_type, charset, payload


class GoogleVisionWebDetectionProvider:
    """Reverse-image search through Google Cloud Vision Web Detection."""

    def __init__(
        self,
        max_results: int = 10,
        *,
        client: Any = None,
        vision_module: Any = None,
    ) -> None:
        self.max_results = int(max_results)
        if self.max_results < 1:
            raise ValueError("max_results must be at least 1")
        self.client = client
        self.vision = vision_module

    def search(self, image_path: Any) -> Dict[str, Any]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError("Image file does not exist: %s" % path)
        self._load()
        image = self.vision.Image(content=path.read_bytes())
        response = self.client.web_detection(image=image)
        error = getattr(getattr(response, "error", None), "message", "")
        if error:
            raise RuntimeError("Google Vision Web Detection failed: %s" % error)
        annotation = getattr(response, "web_detection", None)
        if annotation is None:
            return {"results": [], "entities": [], "best_guess_labels": []}

        results = []
        seen = set()
        for page in getattr(annotation, "pages_with_matching_images", []) or []:
            url = str(getattr(page, "url", "")).strip()
            if not url or url in seen:
                continue
            seen.add(url)
            matching_images = []
            for field in ("full_matching_images", "partial_matching_images"):
                for item in getattr(page, field, []) or []:
                    image_url = str(getattr(item, "url", "")).strip()
                    if image_url and image_url not in matching_images:
                        matching_images.append(image_url)
            results.append(
                {
                    "url": url,
                    "title": str(getattr(page, "page_title", "")).strip(),
                    "matching_image_urls": matching_images[:10],
                }
            )
            if len(results) >= self.max_results:
                break
        entities = [
            {
                "description": str(getattr(item, "description", "")).strip(),
                "score": float(getattr(item, "score", 0.0) or 0.0),
            }
            for item in (getattr(annotation, "web_entities", []) or [])
            if str(getattr(item, "description", "")).strip()
        ]
        labels = [
            str(getattr(item, "label", "")).strip()
            for item in (getattr(annotation, "best_guess_labels", []) or [])
            if str(getattr(item, "label", "")).strip()
        ]
        return {
            "results": results,
            "entities": entities,
            "best_guess_labels": labels,
        }

    def _load(self) -> None:
        if self.vision is None:
            try:
                from google.cloud import vision
            except ImportError as exc:
                raise RuntimeError(
                    "Google Vision provenance requires requirements-provenance.txt"
                ) from exc
            self.vision = vision
        if self.client is None:
            self.client = self.vision.ImageAnnotatorClient()


class ProvenanceRetriever:
    """Turn RIS pages into source-grounded provenance facts."""

    def __init__(self, provider: Any, crawler: WebCrawler, llm: Any, top_k: int = 5):
        self.provider = provider
        self.crawler = crawler
        self.llm = llm
        self.top_k = int(top_k)
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")

    def search_image(
        self,
        image_path: Any,
        evidence_id: str = "P1",
        image_evidence_id: str = "I1",
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "evidence_id": evidence_id,
            "image_evidence_id": image_evidence_id,
            "image_path": str(image_path),
            "search_metadata": {},
            "sources": [],
            "facts": [],
            "first_seen": "",
            "location": "",
            "event": "",
            "people": [],
            "raw_output": None,
            "errors": [],
        }
        errors: List[str] = result["errors"]
        try:
            search = self.provider.search(image_path)
        except Exception as exc:
            errors.append("RIS failed: %s: %s" % (type(exc).__name__, exc))
            return result
        if not isinstance(search, Mapping):
            errors.append("RIS result was not an object")
            return result
        result["search_metadata"] = {
            "entities": search.get("entities", []),
            "best_guess_labels": search.get("best_guess_labels", []),
        }
        candidates = search.get("results", [])
        candidates = candidates if isinstance(candidates, list) else []
        seen = set()
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            url = str(candidate.get("url", "")).strip()
            if not url or url in seen:
                continue
            seen.add(url)
            crawled = self.crawler.crawl(url)
            if not isinstance(crawled, Mapping):
                crawled = {
                    "url": url,
                    "errors": ["Web crawler result was not an object"],
                }
            source_id = "%s.S%d" % (evidence_id, len(result["sources"]) + 1)
            source = {
                "id": source_id,
                "url": str(crawled.get("url", url)),
                "title": str(crawled.get("title") or candidate.get("title", "")),
                "date": str(crawled.get("date", "")),
                "caption": str(crawled.get("caption", "")),
                "text": str(crawled.get("text", "")),
                "matching_image_urls": candidate.get("matching_image_urls", []),
                "errors": list(crawled.get("errors", [])),
            }
            result["sources"].append(source)
            for crawl_error in source["errors"]:
                errors.append("%s: %s" % (source_id, crawl_error))
            if len(result["sources"]) >= self.top_k:
                break

        usable_sources = [
            {
                "id": item["id"],
                "url": item["url"],
                "title": item["title"],
                "date": item["date"],
                "caption": item["caption"],
                "text": item["text"],
            }
            for item in result["sources"]
            if item["title"] or item["date"] or item["caption"] or item["text"]
        ]
        if not usable_sources:
            return result
        model_input = {
            "image_evidence_id": image_evidence_id,
            "search_metadata": result["search_metadata"],
            "pages": usable_sources,
        }
        prompt = render_prompt(
            PROVENANCE_SYSTEM,
            PROVENANCE_USER,
            PROVENANCE_INPUT_JSON=json.dumps(
                model_input, ensure_ascii=False, indent=2
            ),
        )
        try:
            raw_output = self.llm.generate(prompt)
            result["raw_output"] = raw_output
            parsed = parse_json_output(raw_output)
        except Exception as exc:
            errors.append(
                "Provenance extraction failed: %s: %s" % (type(exc).__name__, exc)
            )
            return result

        for field in ("first_seen", "location", "event"):
            value = parsed.get(field, "")
            if isinstance(value, str):
                result[field] = value.strip()
            elif value is not None:
                errors.append("Provenance field %r must be a string" % field)
        people = parsed.get("people", [])
        if isinstance(people, list):
            result["people"] = [
                person.strip()
                for person in people
                if isinstance(person, str) and person.strip()
            ]
        else:
            errors.append("Provenance field 'people' must be a list")
        allowed_sources = {item["id"] for item in usable_sources}
        facts = parsed.get("facts", [])
        if not isinstance(facts, list):
            errors.append("Provenance field 'facts' must be a list")
            return result
        seen_facts = set()
        for index, fact in enumerate(facts[:12]):
            if not isinstance(fact, Mapping):
                errors.append("Provenance fact at index %d must be an object" % index)
                continue
            text = str(fact.get("text", "")).strip()
            source_ids = fact.get("source_ids", [])
            if not text or not isinstance(source_ids, list):
                errors.append(
                    "Provenance fact at index %d requires text and source_ids" % index
                )
                continue
            valid_sources = []
            for source_id in source_ids:
                source_id = str(source_id).strip()
                if source_id not in allowed_sources:
                    errors.append(
                        "Provenance fact at index %d references unknown source %r"
                        % (index, source_id)
                    )
                elif source_id not in valid_sources:
                    valid_sources.append(source_id)
            if not valid_sources or text in seen_facts:
                if not valid_sources:
                    errors.append(
                        "Provenance fact at index %d has no valid source" % index
                    )
                continue
            seen_facts.add(text)
            result["facts"].append(
                {
                    "id": "%s.F%d" % (evidence_id, len(result["facts"]) + 1),
                    "text": text,
                    "source_ids": valid_sources,
                }
            )
        return result


__all__ = [
    "GoogleVisionWebDetectionProvider",
    "ProvenanceRetriever",
    "WebCrawler",
]
