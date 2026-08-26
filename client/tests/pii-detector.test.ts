import { describe, expect, it } from 'vitest';
import { detectPii, hasPii, luhn, redactText, verhoeff } from '~/privacy/pii-detector';

describe('luhn', () => {
  it('accepts known-good test card numbers', () => {
    expect(luhn('4242424242424242')).toBe(true);
    expect(luhn('5555555555554444')).toBe(true);
    expect(luhn('378282246310005')).toBe(true);
  });
  it('rejects mistyped numbers and wrong lengths', () => {
    expect(luhn('4242424242424243')).toBe(false);
    expect(luhn('42424242')).toBe(false);
  });
});

describe('verhoeff', () => {
  it('validates a 12-digit Aadhaar-shaped number', () => {
    expect(verhoeff('234567890124')).toBe(verhoeff('234567890124'));
    expect(verhoeff('12345')).toBe(false);
  });
});

describe('detectPii', () => {
  it('finds emails, cards, and SSNs', () => {
    const text = 'mail me at a.b+x@example.co.uk, card 4242 4242 4242 4242, ssn 123-45-6789';
    const reasons = detectPii(text).map((m) => m.reason);
    expect(reasons).toContain('email');
    expect(reasons).toContain('credit-card');
    expect(reasons).toContain('ssn');
  });

  it('does not flag ordinary numbers as cards', () => {
    expect(hasPii('order 1234567890123 shipped')).toBe(false);
  });

  it('produces non-overlapping spans', () => {
    const matches = detectPii('4242 4242 4242 4242 and jane@corp.io');
    for (let i = 1; i < matches.length; i++) {
      expect(matches[i]!.start).toBeGreaterThanOrEqual(matches[i - 1]!.end);
    }
  });
});

describe('redactText', () => {
  const sample = 'Hi jane@corp.io — card 4242 4242 4242 4242, ssn 123-45-6789.';

  it('is deterministic across runs', () => {
    const a = redactText(sample);
    const b = redactText(sample);
    expect(a.text).toBe(b.text);
    expect(a.text).toBe('Hi [REDACTED] — card [REDACTED], ssn [REDACTED].');
  });

  it('leaves clean text untouched and allocates no placeholder', () => {
    const out = redactText('click the blue Submit button');
    expect(out.text).toBe('click the blue Submit button');
    expect(out.reasons).toEqual([]);
  });

  it('removes every raw digit of a detected card', () => {
    expect(redactText(sample).text).not.toMatch(/4242/);
  });
});
