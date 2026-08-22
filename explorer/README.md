# Local trajectory explorer

Python 3.10+ is required. Install the local dependencies once, then start the
indexer and web server with one command:

```bash
python -m pip install -r requirements.txt
python -m explorer
```

Open <http://127.0.0.1:8000>. The process reads only local `export/*.jsonl` and
`results/*.jsonl` files. It makes no telemetry, analytics, CDN, font, or other
network requests.

For frontend development:

```bash
cd explorer/frontend
npm install
npm run build
```

The production bundle is committed under `explorer/static/`, so Node is not
needed to run the explorer. Dataset rows are never included in that bundle.

## Current checkpoint

The indexer and virtualized Calls view are implemented. Filters, grouping,
timeline, delta, distributions, and their screenshots follow in the remaining
build checkpoints.
