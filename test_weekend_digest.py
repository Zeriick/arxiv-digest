import unittest
from datetime import date, datetime
from types import SimpleNamespace
from urllib.error import HTTPError
from unittest.mock import patch
from zoneinfo import ZoneInfo

from digest_pipeline import main


class WeekendDigestTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "dry_run": False,
            "local_timezone": "Asia/Shanghai",
            "max_selected_papers": 10,
        }
        self.smtp_config = {"to": "test@example.com"}
        self.patchers = [
            patch("digest_pipeline.setup_logging"),
            patch("digest_pipeline.get_runtime_config", return_value=self.config),
            patch("digest_pipeline.get_smtp_config", return_value=self.smtp_config),
            patch("digest_pipeline.validate_runtime_config"),
            patch("digest_pipeline.log_runtime_config"),
            patch("digest_pipeline.load_seen", return_value=set()),
            patch("digest_pipeline.load_openalex_cache", return_value={}),
            patch("digest_pipeline.is_local_weekend", return_value=True),
            patch("digest_pipeline.write_text_artifact", return_value="email_preview.html"),
            patch("digest_pipeline.write_json_artifact", return_value="pipeline_summary.json"),
            patch("digest_pipeline.get_run_dir", return_value=None),
            patch("digest_pipeline.save_seen"),
            patch("digest_pipeline.save_openalex_cache"),
            patch("digest_pipeline.send_email"),
        ]
        self.mocks = [patcher.start() for patcher in self.patchers]
        self.addCleanup(self.stop_patchers)

    def stop_patchers(self):
        for patcher in reversed(self.patchers):
            patcher.stop()

    @patch("digest_pipeline.fetch_papers")
    def test_weekend_406_sends_source_unavailable_email_and_succeeds(self, fetch_mock):
        fetch_mock.side_effect = HTTPError(
            "https://export.arxiv.org/api/query", 406, "Not Acceptable", {}, None
        )

        main()

        from digest_pipeline import send_email, write_json_artifact, save_seen

        self.assertIn("暂时没能获取 arXiv", send_email.call_args.args[0])
        self.assertEqual(send_email.call_args.kwargs["subject"], "论文日报暂无更新｜周末愉快")
        self.assertEqual(write_json_artifact.call_args.args[1]["fetch_status"], "source_unavailable")
        save_seen.assert_not_called()

    @patch("digest_pipeline.fetch_papers")
    def test_weekend_empty_feed_sends_no_papers_email(self, fetch_mock):
        target = {"label_date": date(2026, 9, 27)}
        fetch_mock.return_value = ([], target, 1)

        main()

        from digest_pipeline import send_email, write_json_artifact

        self.assertIn("今天没有新论文", send_email.call_args.args[0])
        self.assertEqual(write_json_artifact.call_args.args[1]["fetch_status"], "no_papers")

    @patch("digest_pipeline.fetch_papers")
    def test_dry_run_does_not_send_weekend_email(self, fetch_mock):
        self.config["dry_run"] = True
        fetch_mock.side_effect = HTTPError(
            "https://export.arxiv.org/api/query", 406, "Not Acceptable", {}, None
        )

        main()

        from digest_pipeline import send_email

        send_email.assert_not_called()

    @patch("digest_pipeline.fetch_papers")
    def test_weekday_406_still_fails(self, fetch_mock):
        from digest_pipeline import is_local_weekend, send_email

        is_local_weekend.return_value = False
        fetch_mock.side_effect = HTTPError(
            "https://export.arxiv.org/api/query", 406, "Not Acceptable", {}, None
        )

        with self.assertRaises(HTTPError):
            main()

        send_email.assert_not_called()

    @patch("digest_pipeline.fetch_papers")
    def test_weekend_nonretryable_http_error_still_fails(self, fetch_mock):
        from digest_pipeline import send_email

        fetch_mock.side_effect = HTTPError(
            "https://export.arxiv.org/api/query", 400, "Bad Request", {}, None
        )

        with self.assertRaises(HTTPError):
            main()

        send_email.assert_not_called()

    @patch("digest_pipeline.batch_assess_papers", return_value=[])
    @patch("digest_pipeline.fetch_papers")
    def test_weekend_all_papers_seen_sends_empty_email(self, fetch_mock, assess_mock):
        from digest_pipeline import load_seen, send_email, write_json_artifact

        paper_id = "http://arxiv.org/abs/2609.12345"
        load_seen.return_value = {paper_id}
        paper = SimpleNamespace(
            id=paper_id,
            title="Previously Seen Paper",
            summary="An abstract",
            link="https://arxiv.org/abs/2609.12345",
            authors=[],
        )
        announcement_et = datetime(2026, 9, 27, 20, tzinfo=ZoneInfo("America/New_York"))
        target = {
            "label_date": date(2026, 9, 28),
            "announcement_et": announcement_et,
            "announcement_local": announcement_et.astimezone(ZoneInfo("Asia/Shanghai")),
        }
        fetch_mock.return_value = ([paper], target, 1)

        main()

        assess_mock.assert_called_once_with([], self.config)
        self.assertIn("今天没有新论文", send_email.call_args.args[0])
        self.assertEqual(write_json_artifact.call_args.args[1]["stats"]["skipped_seen"], 1)


if __name__ == "__main__":
    unittest.main()
