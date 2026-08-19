import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import feedparser

from digest_sources import (
    fetch_papers_from_current_rss,
    get_arxiv_announcement_for_rss_entry,
    normalize_arxiv_rss_entry,
)


ATOM_FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:dc="http://purl.org/dc/elements/1.1/">
  <id>http://rss.arxiv.org/atom/cs.OS+cs.PL</id>
  <title>Test arXiv updates</title>
  <updated>2026-08-19T04:05:00+00:00</updated>
  <entry>
    <id>oai:arXiv.org:2608.12345v1</id>
    <title>A Useful Systems Paper</title>
    <updated>2026-08-19T04:05:00+00:00</updated>
    <link href="https://arxiv.org/abs/2608.12345" rel="alternate" type="text/html"/>
    <summary>arXiv:2608.12345v1 Announce Type: new
Abstract: The actual abstract.</summary>
    <category term="cs.OS"/>
    <published>2026-08-19T00:00:00-04:00</published>
    <arxiv:announce_type>new</arxiv:announce_type>
    <dc:creator>Ada Lovelace, Grace Hopper</dc:creator>
  </entry>
  <entry>
    <id>oai:arXiv.org:2501.54321v2</id>
    <title>An Updated Paper</title>
    <updated>2026-08-19T04:05:00+00:00</updated>
    <link href="https://arxiv.org/abs/2501.54321" rel="alternate" type="text/html"/>
    <summary>arXiv:2501.54321v2 Announce Type: replace
Abstract: A replacement abstract.</summary>
    <category term="cs.PL"/>
    <published>2026-08-19T00:00:00-04:00</published>
    <arxiv:announce_type>replace</arxiv:announce_type>
    <dc:creator>Barbara Liskov</dc:creator>
  </entry>
</feed>
"""


def parse_fixture():
    feed = feedparser.parse(ATOM_FIXTURE)
    if feed.bozo:
        raise AssertionError(feed.bozo_exception)
    return feed


def target_announcement(day=18):
    et = ZoneInfo("America/New_York")
    local = ZoneInfo("Asia/Shanghai")
    announcement_et = datetime(2026, 8, day, 20, 0, tzinfo=et)
    return {
        "announcement_et": announcement_et,
        "announcement_local": announcement_et.astimezone(local),
        "label_date": announcement_et.astimezone(local).date(),
    }


class ArxivRssTests(unittest.TestCase):
    def test_rss_midnight_maps_to_previous_evening_announcement(self):
        entry = parse_fixture().entries[0]

        announcement = get_arxiv_announcement_for_rss_entry(entry)

        self.assertEqual(
            announcement,
            datetime(2026, 8, 18, 20, 0, tzinfo=ZoneInfo("America/New_York")),
        )

    def test_normalize_rss_entry_matches_search_api_shape(self):
        entry = normalize_arxiv_rss_entry(parse_fixture().entries[0])

        self.assertEqual(entry.id, "http://arxiv.org/abs/2608.12345v1")
        self.assertEqual(entry.link, "https://arxiv.org/abs/2608.12345v1")
        self.assertEqual(entry.summary, "The actual abstract.")
        self.assertEqual(
            [author["name"] for author in entry.authors],
            ["Ada Lovelace", "Grace Hopper"],
        )

    @patch("digest_sources.write_json_artifact")
    @patch("digest_sources.fetch_arxiv_feed")
    def test_current_feed_selects_new_entries_and_skips_replacements(
        self,
        fetch_mock,
        write_artifact_mock,
    ):
        fetch_mock.return_value = parse_fixture()
        config = {"local_timezone": "Asia/Shanghai", "target_days_ago": 1}

        papers, target, pages = fetch_papers_from_current_rss(
            config,
            target_announcement(),
        )

        self.assertEqual([paper.id for paper in papers], ["http://arxiv.org/abs/2608.12345v1"])
        self.assertEqual(target, target_announcement())
        self.assertEqual(pages, 1)
        artifact_entries = write_artifact_mock.call_args.args[1]
        self.assertEqual(len(artifact_entries), 1)
        self.assertEqual(artifact_entries[0]["announce_type"], "new")
        self.assertEqual(artifact_entries[0]["source"], "arxiv_atom_feed")

    @patch("digest_sources.write_json_artifact")
    @patch("digest_sources.fetch_arxiv_feed")
    def test_stale_feed_defers_to_search_api(
        self,
        fetch_mock,
        write_artifact_mock,
    ):
        fetch_mock.return_value = parse_fixture()
        config = {"local_timezone": "Asia/Shanghai", "target_days_ago": 1}

        result = fetch_papers_from_current_rss(
            config,
            target_announcement(day=19),
        )

        self.assertIsNone(result)
        write_artifact_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
