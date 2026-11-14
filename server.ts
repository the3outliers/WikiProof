/**
 * Development/preview launcher for WikiProof in AI Studio.
 * Bridges `npm run dev` (port 3000) directly to the FastAPI Uvicorn server
 * serving `frontend/index.html` and `/api/*` endpoints.
 */

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';

function getArg(flag: string, fallback: string): string {
  const eqArg = process.argv.find((a) => a.startsWith(`${flag}=`));
  if (eqArg) return eqArg.slice(flag.length + 1);
  const idx = process.argv.indexOf(flag);
  if (idx !== -1 && process.argv[idx + 1]) return process.argv[idx + 1];
  return fallback;
}

const port = getArg('--port', process.env.PORT || '3000');
const host = getArg('--host', '0.0.0.0');

const venvPython = path.resolve(process.cwd(), '.venv/bin/python');
const pythonBin = existsSync(venvPython) ? venvPython : 'python3';

const child = spawn(
  pythonBin,
  ['-m', 'uvicorn', 'main:app', '--host', host, '--port', port],
  {
    stdio: 'inherit',
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
    },
  },
);

const forwardSignal = (signal: NodeJS.Signals) => {
  if (!child.killed) {
    child.kill(signal);
  }
};

process.on('SIGINT', () => forwardSignal('SIGINT'));
process.on('SIGTERM', () => forwardSignal('SIGTERM'));

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
