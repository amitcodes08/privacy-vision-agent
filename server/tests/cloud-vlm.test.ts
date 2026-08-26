import { describe, expect, it } from 'vitest';
import { extractJson, heuristicPlan } from '../src/ai/cloud-vlm.ts';
import type { InferenceRequestPayload, ScrubbedDom } from '../../shared/types.ts';

const dom: ScrubbedDom = {
  url: 'https://shop.test/checkout',
  origin: 'https://shop.test',
  title: 'Checkout',
  viewport: { width: 1280, height: 800, scrollX: 0, scrollY: 0 },
  nodes: [
    { id: 0, tag: 'button', selector: '#accept-cookies', text: 'Accept cookies', visible: true },
    { id: 1, tag: 'input', type: 'email', selector: '#email', label: 'Email address', visible: true },
    { id: 2, tag: 'button', selector: '#pay', text: 'Pay now', visible: true, disabled: true },
  ],
  redactionSummary: {},
};

const request = (goal: string): InferenceRequestPayload => ({
  goal,
  imageBase64: 'x'.repeat(128),
  imageMime: 'image/jpeg',
  dom,
});

describe('heuristicPlan', () => {
  it('clicks the element whose text matches the goal', () => {
    const decision = heuristicPlan(request('accept the cookies banner'));
    expect(decision.action).toMatchObject({ action: 'click', selector: '#accept-cookies' });
    expect(decision.source).toBe('heuristic');
  });

  it('fills email inputs with a value token, never a literal', () => {
    const decision = heuristicPlan(request('enter my email address'));
    expect(decision.action).toMatchObject({ action: 'fill', selector: '#email', valueType: 'USER_EMAIL' });
    expect(JSON.stringify(decision.action)).not.toContain('value"');
  });

  it('ignores disabled elements', () => {
    const decision = heuristicPlan(request('pay now'));
    expect(JSON.stringify(decision.action)).not.toContain('#pay');
  });

  it('returns done with low confidence when nothing matches', () => {
    const decision = heuristicPlan(request('zzzz qqqq'));
    expect(decision.action.action).toBe('done');
    expect(decision.confidence).toBeLessThan(0.5);
  });
});

describe('extractJson', () => {
  it('pulls the first balanced object out of chatty output', () => {
    expect(extractJson('sure! {"action":"click","selector":"#a"} hope that helps')).toEqual({
      action: 'click',
      selector: '#a',
    });
  });

  it('handles nesting and returns null on garbage', () => {
    expect(extractJson('{"a":{"b":1}}')).toEqual({ a: { b: 1 } });
    expect(extractJson('no json here')).toBeNull();
    expect(extractJson('{"broken":')).toBeNull();
  });
});

describe('sanitize', () => {
  it('blocks password and OTP fills', async () => {
    const { sanitize } = await import('../src/ai/cloud-vlm.ts');
    expect(sanitize({ action: 'fill', selector: '#pw', valueType: 'USER_PASSWORD' })).toEqual({
      action: 'done',
      summary: 'blocked: server attempted a credential fill',
    });
    expect(sanitize({ action: 'fill', selector: '#otp', valueType: 'OTP_CODE' })).toEqual({
      action: 'done',
      summary: 'blocked: server attempted a credential fill',
    });
  });

  it('drops literal value property when valueType is a token', async () => {
    const { sanitize } = await import('../src/ai/cloud-vlm.ts');
    const sanitized = sanitize({
      action: 'fill',
      selector: '#email',
      valueType: 'USER_EMAIL',
      value: 'attacker@evil.com',
    } as any);
    expect(sanitized).toEqual({
      action: 'fill',
      selector: '#email',
      valueType: 'USER_EMAIL',
    });
  });
});

