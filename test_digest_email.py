import unittest

from digest_email import DEFAULT_EMAIL_SUBJECT, build_email_subject


class EmailSubjectTests(unittest.TestCase):
    def test_subject_stays_unchanged_without_90_plus_papers(self):
        papers = [{"score": 89}, {"score": 80}]

        self.assertEqual(build_email_subject(papers), DEFAULT_EMAIL_SUBJECT)

    def test_score_of_exactly_90_triggers_high_score_alert(self):
        papers = [{"score": 90}]

        self.assertEqual(
            build_email_subject(papers),
            f"🚨 必读：1 篇 90+ 高分论文（最高 90 分）｜{DEFAULT_EMAIL_SUBJECT}",
        )

    def test_alert_reports_high_score_count_and_maximum(self):
        papers = [{"score": 96}, {"score": 91}, {"score": 89}]

        self.assertEqual(
            build_email_subject(papers),
            f"🚨 必读：2 篇 90+ 高分论文（最高 96 分）｜{DEFAULT_EMAIL_SUBJECT}",
        )


if __name__ == "__main__":
    unittest.main()
