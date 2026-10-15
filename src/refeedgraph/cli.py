"""Command-line entry points for graph construction and feedback generation."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
from .common import (load_json, write_json, load_panelbench_debate,
                     format_debate_for_prompt, make_client)
from .method import (load_graph, generate_generic_issues, retrieve_graph_context,
                     specify_our_feedback, revise_debate_speech, speech_for_side)


def validate_debate(debate):
    if not isinstance(debate, dict) or not debate.get('dataset') or not debate.get('debate_id'):
        raise ValueError('Debate requires dataset and debate_id.')
    motion = debate.get('motion', {})
    if not isinstance(motion, dict) or not motion.get('motion'):
        raise ValueError('Debate requires motion.motion.')
    speeches = debate.get('speeches')
    if not isinstance(speeches, list) or not speeches:
        raise ValueError('Debate requires a nonempty speeches list.')
    if any(not s.get('debater_name') or not str(s.get('content', '')).strip() for s in speeches):
        raise ValueError('Each speech requires debater_name and nonempty content.')
    return debate


def checked_graph(path):
    graph = load_graph(path)
    if not any(n.get('type') == 'Principle' for n in graph['nodes'].values()):
        raise ValueError('Graph contains no Principle nodes.')
    for e in graph['data'].get('edges', []):
        if e.get('source') not in graph['nodes'] or e.get('target') not in graph['nodes']:
            raise ValueError('Graph contains an edge with a missing endpoint.')
    return graph


def ensure_new(path):
    if path.exists():
        raise FileExistsError(f'Output already exists: {path}. Choose a new output path.')


def model_args(parser):
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--temperature', type=float, default=0.0)


def retrieval_args(parser):
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--retrieval', choices=['lexical', 'embedding'], default='embedding')
    parser.add_argument('--embedding-model', default='text-embedding-3-small')
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--examples-per-principle', type=int, default=2)


def context_for(args, issues):
    if args.top_k < 1 or args.examples_per_principle < 0:
        raise ValueError('top-k must be positive; examples-per-principle must be nonnegative.')
    graph = checked_graph(args.graph)
    embedding_client = None
    if args.retrieval == 'embedding':
        from openai import OpenAI
        key = os.getenv('EMBEDDING_API_KEY') or os.getenv('OPENAI_API_KEY')
        if not key:
            raise ValueError('Set EMBEDDING_API_KEY or OPENAI_API_KEY for embedding retrieval.')
        embedding_client = OpenAI(api_key=key, base_url=os.getenv('EMBEDDING_BASE_URL', 'https://api.openai.com/v1'))
    context = retrieve_graph_context(
        graph, issues, top_k_principles=args.top_k,
        specific_examples_per_principle=args.examples_per_principle,
        retrieval_method=args.retrieval, embedding_model=args.embedding_model,
        client=embedding_client,
    )
    if not context['retrieved_principles']:
        raise ValueError('No principles retrieved. Check graph issue labels and ADDRESSES_ISSUE edges.')
    return context


def build(args):
    ensure_new(args.output_dir)
    debates = [validate_debate(load_json(p)) for p in args.inputs]
    ids = [(d['dataset'], d['debate_id']) for d in debates]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate training debate IDs.')
    # The original graph merger namespaces by debate_id, so require unique IDs across datasets too.
    if len({d['debate_id'] for d in debates}) != len(debates):
        raise ValueError('Build graphs separately per dataset; debate IDs must be unique.')
    args.output_dir.mkdir(parents=True)
    graphs = []
    for index, path in enumerate(args.inputs):
        folder = args.output_dir / f'{index:04d}'
        folder.mkdir()
        files = {name: folder / f'{name}.json' for name in ['judge', 'classify', 'deduplicate', 'abstract', 'graph']}
        calls = [
            ('judge', ['--input', str(path.resolve()), '--output', str(files['judge'])]),
            ('classify', ['--input', str(files['judge']), '--output', str(files['classify'])]),
            ('deduplicate', ['--input', str(files['classify']), '--output', str(files['deduplicate'])]),
            ('abstract', ['--input', str(files['deduplicate']), '--output', str(files['abstract'])]),
            ('build', ['--step2', str(files['classify']), '--step2b', str(files['deduplicate']), '--step3', str(files['abstract']), '--output', str(files['graph'])]),
        ]
        for module, flags in calls:
            if module != 'build':
                flags += ['--model', args.model, '--temperature', str(args.temperature)]
                if args.mock:
                    flags += ['--mock']
            subprocess.run([sys.executable, '-m', f'refeedgraph.stages.{module}', *flags], check=True)
        graphs.append(files['graph'])
    combined = args.output_dir / 'graph.json'
    subprocess.run([sys.executable, '-m', 'refeedgraph.stages.combine', '--inputs', *map(str, graphs), '--output', str(combined)], check=True)
    checked_graph(combined)
    write_json(args.output_dir / 'run.json', {'model': args.model, 'temperature': args.temperature,
        'mock': args.mock, 'training_debates': [{'dataset': d['dataset'], 'debate_id': d['debate_id']} for d in debates]})
    print(f'Graph saved to {combined}' + (' (MOCK: plumbing test only)' if args.mock else ''))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build', help='Build a training graph from normalized debate JSON files')
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--mock', action='store_true', help='Use synthetic stage responses; no model API calls')
    model_args(p)
    p = sub.add_parser('export', help='Convert locally obtained dataset files to normalized debate JSON')
    p.add_argument('--dataset', choices=['debateart', 'iq2', 'bp_og_oo'], required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--id')
    group.add_argument('--split', type=Path)
    p.add_argument('--partition', choices=['train', 'test'], default='train')
    p.add_argument('--output-dir', type=Path, required=True)
    p = sub.add_parser('retrieve', help='Retrieve graph evidence for saved issues; lexical mode is offline')
    p.add_argument('--issues', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    retrieval_args(p)
    p = sub.add_parser('generate', help='Diagnose a debate and generate graph-grounded feedback')
    p.add_argument('--debate', type=Path, required=True)
    p.add_argument('--target', required=True, help='Exact debater_name to receive feedback')
    p.add_argument('--variant', choices=['principles', 'specific'], default='specific')
    p.add_argument('--output', type=Path, required=True)
    model_args(p); retrieval_args(p)
    p = sub.add_parser('rewrite', help='Apply generated feedback to the target speech')
    p.add_argument('--debate', type=Path, required=True)
    p.add_argument('--feedback', type=Path, required=True, help='JSON saved by generate')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-tokens', type=int, default=10000)
    model_args(p)
    args = parser.parse_args(argv)
    if args.command == 'build':
        return build(args)
    if args.command == 'export':
        ensure_new(args.output_dir)
        rows = [{'id': args.id}] if args.id else load_json(args.split)[args.partition]
        debates = [validate_debate(load_panelbench_debate(args.dataset, str(row['id']))) for row in rows]
        for debate in debates:
            write_json(args.output_dir / f"{args.dataset}_{debate['debate_id']}.json", debate)
        return
    ensure_new(args.output)
    if args.command == 'retrieve':
        issues = load_json(args.issues)
        if isinstance(issues, dict):
            issues = issues['issues']
        write_json(args.output, context_for(args, issues))
        return
    debate = validate_debate(load_json(args.debate))
    text = format_debate_for_prompt(debate)
    if args.command == 'generate':
        speech_for_side(debate, args.target)
        checked_graph(args.graph)
        client = make_client()
        issues = generate_generic_issues(client, model=args.model, debate_text=text,
                                        losing_side=args.target, temperature=args.temperature)
        context = context_for(args, issues)
        examples = context['specific_feedback'] if args.variant == 'specific' else []
        feedback = specify_our_feedback(client, model=args.model, debate_text=text, losing_side=args.target,
            issues=issues, principles=context['retrieved_principles'], supporting_examples=examples,
            temperature=args.temperature)
        if not feedback.strip():
            raise ValueError('Model returned empty feedback.')
        write_json(args.output, {'dataset': debate['dataset'], 'debate_id': debate['debate_id'],
            'target': args.target, 'variant': args.variant, 'model': args.model,
            'temperature': args.temperature, 'retrieval': args.retrieval,
            'embedding_model': args.embedding_model if args.retrieval == 'embedding' else None,
            'issues': issues, 'context': context, 'feedback': feedback})
    else:
        feedback = load_json(args.feedback)
        if (feedback['dataset'], feedback['debate_id']) != (debate['dataset'], debate['debate_id']):
            raise ValueError('Feedback and debate identifiers do not match.')
        original = speech_for_side(debate, feedback['target'])
        revised = revise_debate_speech(make_client(), model=args.model, title=debate['motion']['motion'],
            debate_text=text, losing_side=feedback['target'], original_speech=original,
            feedback=feedback['feedback'], temperature=args.temperature, max_tokens=args.max_tokens)
        write_json(args.output, {'dataset': debate['dataset'], 'debate_id': debate['debate_id'],
            'target': feedback['target'], 'model': args.model, 'temperature': args.temperature,
            'original_speech': original, 'revised_speech': revised})

if __name__ == '__main__':
    main()
