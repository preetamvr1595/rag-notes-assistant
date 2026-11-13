"""Command line interface for indexing, serving, and query testing."""
# Enhanced with stats and serve subcommands
"""Command line interface for indexing and query testing."""
"""Command line:  python -m rag_notes.cli build   |   python -m rag_notes.cli ask "How do I merge two DataFrames?" """
import argparse, sys
from .answer import RAG, backend_from_env
from .index import Index

INDEX_PATH = "models/index.joblib"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rag_notes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="read data/notes and build the search index")
    b.add_argument("--notes", default="data/notes"); b.add_argument("--minilm", action="store_true", help="also add neural embeddings (needs sentence-transformers + internet)")
    a = sub.add_parser("ask", help="ask a question"); a.add_argument("question"); a.add_argument("--series", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        idx = Index.build(args.notes, use_minilm=args.minilm); idx.save(INDEX_PATH)
        print(f"indexed {len(idx.chunks)} chunks -> {INDEX_PATH}")
    else:
        ans = RAG(Index.load(INDEX_PATH), backend_from_env()).ask(args.question, series=args.series)
        print(ans.text, "\n")
        print(f"[backend={ans.backend}  coverage={ans.confidence:.2f}  answered={ans.answered}]")
        for s in ans.sources:
            print(f"  [{s['n']}] {s['source']} › {s['section']}")


if __name__ == "__main__":
    sys.exit(main())
