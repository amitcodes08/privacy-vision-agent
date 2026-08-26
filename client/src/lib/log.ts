import type { AgentLogEntry } from '@shared/types';

const RING_SIZE = 200;
const ring: AgentLogEntry[] = [];
const listeners = new Set<(e: AgentLogEntry) => void>();

export function log(
  level: AgentLogEntry['level'],
  scope: string,
  message: string,
  data?: unknown,
): AgentLogEntry {
  const entry: AgentLogEntry = { ts: Date.now(), level, scope, message, data };
  ring.push(entry);
  if (ring.length > RING_SIZE) ring.shift();
  const line = `[pva:${scope}] ${message}`;
  if (level === 'error') console.error(line, data ?? '');
  else if (level === 'warn') console.warn(line, data ?? '');
  else console.log(line, data ?? '');
  for (const fn of listeners) fn(entry);
  return entry;
}

export const logger = {
  debug: (scope: string, m: string, d?: unknown) => log('debug', scope, m, d),
  info: (scope: string, m: string, d?: unknown) => log('info', scope, m, d),
  warn: (scope: string, m: string, d?: unknown) => log('warn', scope, m, d),
  error: (scope: string, m: string, d?: unknown) => log('error', scope, m, d),
};

export const recentLogs = (n = 60): AgentLogEntry[] => ring.slice(-n);
export const onLog = (fn: (e: AgentLogEntry) => void) => {
  listeners.add(fn);
  return () => listeners.delete(fn);
};
