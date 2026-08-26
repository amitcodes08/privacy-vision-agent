/**
 * Deterministic PII detection. Pure functions only — no DOM access here so
 * the rules can be unit-tested in isolation and reused by the redactor.
 */
import type { RedactionReason } from '@shared/types';

export interface PiiMatch {
  reason: RedactionReason;
  start: number;
  end: number;
  raw: string;
}

/** Ordered: earlier rules win when spans overlap. */
const RULES: ReadonlyArray<{ reason: RedactionReason; re: RegExp; validate?: (s: string) => boolean }> = [
  // 13-19 digits, optionally grouped by spaces/dashes, Luhn-checked.
  {
    reason: 'credit-card',
    re: /\b(?:\d[ -]?){12,18}\d\b/g,
    validate: (s) => luhn(s.replace(/\D/g, '')),
  },
  { reason: 'ssn', re: /\b(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b/g },
  // Aadhaar: 12 digits starting 2-9, Verhoeff-checked to cut false hits.
  { reason: 'aadhaar', re: /\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b/g, validate: (s) => verhoeff(s.replace(/\D/g, '')) },
  { reason: 'email', re: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g },
  // E.164-ish and common national formats; require a separator or + to
  // avoid swallowing order numbers.
  {
    reason: 'phone',
    re: /(?:\+\d{1,3}[ -]?)?(?:\(\d{2,4}\)[ -]?|\d{3,5}[ -])\d{3}[ -]?\d{3,4}\b/g,
  },
  { reason: 'otp', re: /\b(?:otp|one[- ]time (?:code|password)|verification code)\D{0,12}(\d{4,8})\b/gi },
];

export function luhn(digits: string): boolean {
  if (digits.length < 13 || digits.length > 19) return false;
  let sum = 0;
  let double = false;
  for (let i = digits.length - 1; i >= 0; i--) {
    let d = digits.charCodeAt(i) - 48;
    if (d < 0 || d > 9) return false;
    if (double) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
    double = !double;
  }
  return sum % 10 === 0;
}

const VERHOEFF_D = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
  [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
  [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
  [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
  [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
  [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
  [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
  [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
  [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];
const VERHOEFF_P = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
  [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
  [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
  [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
  [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
  [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
  [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
];

export function verhoeff(digits: string): boolean {
  if (digits.length !== 12) return false;
  let c = 0;
  const rev = digits.split('').reverse();
  for (let i = 0; i < rev.length; i++) {
    const d = Number(rev[i]);
    if (!Number.isInteger(d)) return false;
    c = VERHOEFF_D[c]![VERHOEFF_P[i % 8]![d]!]!;
  }
  return c === 0;
}

/** All PII spans in `text`, de-overlapped, sorted by position. */
export function detectPii(text: string): PiiMatch[] {
  const found: PiiMatch[] = [];
  for (const { reason, re, validate } of RULES) {
    re.lastIndex = 0;
    for (let m = re.exec(text); m !== null; m = re.exec(text)) {
      const raw = m[0];
      if (validate && !validate(raw)) continue;
      found.push({ reason, start: m.index, end: m.index + raw.length, raw });
      if (re.lastIndex === m.index) re.lastIndex++; // guard zero-width
    }
  }
  found.sort((a, b) => a.start - b.start || b.end - a.end);
  const out: PiiMatch[] = [];
  let cursor = -1;
  for (const match of found) {
    if (match.start >= cursor) {
      out.push(match);
      cursor = match.end;
    }
  }
  return out;
}

/**
 * Replace every PII span with `placeholder`. Deterministic: same input
 * always yields the same output (asserted in tests).
 */
export function redactText(
  text: string,
  placeholder = '[REDACTED]',
): { text: string; reasons: RedactionReason[] } {
  const matches = detectPii(text);
  if (matches.length === 0) return { text, reasons: [] };
  let out = '';
  let last = 0;
  for (const m of matches) {
    out += text.slice(last, m.start) + placeholder;
    last = m.end;
  }
  out += text.slice(last);
  return { text: out, reasons: [...new Set(matches.map((m) => m.reason))] };
}

export function hasPii(text: string): boolean {
  return detectPii(text).length > 0;
}
