import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from refeedgraph.cli import checked_graph, main, validate_debate

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples'

class MethodTests(unittest.TestCase):
    def test_retrieval_matches_example_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'retrieved.json'
            main(['retrieve', '--graph', str(EXAMPLES/'graph.json'), '--issues', str(EXAMPLES/'issues.json'), '--retrieval', 'lexical', '--output', str(output)])
            result = json.loads(output.read_text())
            self.assertEqual(result, json.loads((EXAMPLES/'expected_retrieval.json').read_text()))
            self.assertEqual(result['specific_feedback'][0]['principle_id'], result['retrieved_principles'][0]['id'])
            with self.assertRaises(FileExistsError):
                main(['retrieve', '--graph', str(EXAMPLES/'graph.json'), '--issues', str(EXAMPLES/'issues.json'), '--retrieval', 'lexical', '--output', str(output)])

    def test_invalid_graph_is_rejected(self):
        graph = json.loads((EXAMPLES/'graph.json').read_text())
        graph['edges'][0]['target'] = 'missing-node'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'bad.json'; path.write_text(json.dumps(graph))
            with self.assertRaisesRegex(ValueError, 'missing endpoint'):
                checked_graph(path)

    def test_generation_variant_controls_examples(self):
        issues = json.loads((EXAMPLES/'issues.json').read_text())['issues']
        with tempfile.TemporaryDirectory() as folder:
            for variant in ['principles', 'specific']:
                with patch('refeedgraph.cli.make_client'), patch('refeedgraph.cli.generate_generic_issues', return_value=issues), patch('refeedgraph.cli.specify_our_feedback', return_value='Synthetic stub feedback') as generate:
                    output = Path(folder)/f'{variant}.json'
                    main(['generate', '--debate', str(EXAMPLES/'debate.json'), '--graph', str(EXAMPLES/'graph.json'), '--target', 'Pro', '--variant', variant, '--retrieval', 'lexical', '--output', str(output)])
                    self.assertEqual(bool(generate.call_args.kwargs['supporting_examples']), variant=='specific')
                    self.assertEqual(json.loads(output.read_text())['target'], 'Pro')

    def test_mock_construction_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'training'
            env = dict(os.environ)
            for name in ['OPENAI_API_KEY', 'GROQ_API_KEY', 'DEEPINFRA_API_KEY', 'EMBEDDING_API_KEY']:
                env.pop(name, None)
            result = subprocess.run([sys.executable, '-m', 'refeedgraph.cli', 'build', '--inputs', str(EXAMPLES/'debate.json'), '--mock', '--output-dir', str(output)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            graph = checked_graph(output/'graph.json')
            self.assertTrue(graph['nodes'])
            self.assertTrue(json.loads((output/'run.json').read_text())['mock'])

    def test_splits_are_disjoint_and_have_paper_counts(self):
        for dataset, train, test in [('debateart',20,74),('iq2',20,70),('bp_og_oo',7,7)]:
            split = json.loads((ROOT/f'data/splits/{dataset}.json').read_text())
            a={x['id'] for x in split['train']}; b={x['id'] for x in split['test']}
            self.assertEqual((len(a),len(b)), (train,test))
            self.assertFalse(a & b)

    def test_invalid_debate(self):
        with self.assertRaises(ValueError): validate_debate({})

if __name__ == '__main__': unittest.main()
