import json
import unittest
from unittest.mock import patch

from digest_email import build_email
from digest_llm import batch_assess_papers, validate_assessment_payload, validate_summary_payload
from digest_pipeline import build_stats, process_assessment_results, summarize_ranked_candidates


class SelectedPaperExplanationTests(unittest.TestCase):
    def test_compact_assessments_accept_relevant_and_irrelevant_papers(self):
        for relevant, score, area in [(True, 84, "AI-Compiler"), (False, 0, "Irrelevant")]:
            with self.subTest(relevant=relevant):
                payload = {"relevant": relevant, "score": score, "fit_area": area}
                self.assertEqual(validate_assessment_payload(payload), payload)

    def test_legacy_explanations_are_not_carried_into_assessments(self):
        payload = {
            "relevant": True, "score": 84, "fit_area": "AI-Compiler",
            "reason": "Legacy reason", "affiliation_signal": "Legacy signal",
        }
        self.assertEqual(
            set(validate_assessment_payload(payload)), {"relevant", "score", "fit_area"}
        )

    def test_selected_summary_requires_both_email_explanations(self):
        payload = {
            "summary": ["Problem", "Method", "Result"], "translation": "中文概述",
            "reason": "Concrete contribution", "affiliation_signal": "No useful signal",
        }
        for field in ("reason", "affiliation_signal"):
            with self.subTest(field=field):
                incomplete = dict(payload)
                incomplete.pop(field)
                with self.assertRaisesRegex(ValueError, field):
                    validate_summary_payload(incomplete)

    def test_only_top_ten_receive_explanations_and_email_uses_them(self):
        for workers in (1, 2):
            with self.subTest(workers=workers):
                config = {
                    "max_selected_papers": 10,
                    "llm_assess_max_workers": workers,
                    "llm_summary_max_workers": workers,
                }
                papers = [
                    {
                        "id": str(index), "title": f"Paper {index}",
                        "abstract": f"Abstract {index}", "link": f"https://example.org/{index}",
                        "paper_tag": str(index),
                        "authors": [{"name": "Researcher", "affiliation": "Test University"}],
                        "openalex": {},
                    }
                    for index in range(13)
                ]

                def model_response(prompt, stage, tag, runtime_config):
                    if stage == "assess":
                        index = int(tag)
                        return json.dumps({
                            "relevant": index < 12, "score": 95 - index if index < 12 else 0,
                            "fit_area": "AI-Compiler" if index < 12 else "Irrelevant",
                        })
                    self.assertIn("Test University", prompt)
                    return json.dumps({
                        "summary": ["Problem", "Method", "Result"], "translation": "中文概述",
                        "reason": f"Selected contribution {tag}",
                        "affiliation_signal": "Test University is only a confidence signal.",
                    })

                stats = build_stats()
                seen = set()
                assessments = []
                with patch("digest_llm.llm_call", side_effect=model_response) as call:
                    results = batch_assess_papers(papers, config)
                    candidates = process_assessment_results(results, stats, seen, assessments)
                    ranked = sorted(candidates, key=lambda item: -item["score"])
                    selected = summarize_ranked_candidates(ranked, config, stats)

                self.assertEqual([paper["id"] for paper in selected], [str(i) for i in range(10)])
                self.assertEqual(stats["relevance_filtered"], 1)
                self.assertEqual(len(seen), 13)
                self.assertTrue(all("reason" not in item for item in assessments + candidates))
                summary_calls = [item for item in call.call_args_list if item.args[1] == "summary"]
                self.assertEqual(len(summary_calls), 10)
                html = build_email(selected)
                self.assertIn("Selected contribution", html)
                self.assertIn("Test University is only a confidence signal.", html)
                self.assertNotIn("Paper 10</", html)


if __name__ == "__main__":
    unittest.main()
