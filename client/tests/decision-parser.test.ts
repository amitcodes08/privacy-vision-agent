import { describe, expect, it } from 'vitest';
import type { ScrubbedDom, ScrubbedNode } from '@shared/types';
import { extractJson, parseAction } from '~/ai/decision-parser';

const node = (over: Partial<ScrubbedNode> & { id: number; selector: string }): ScrubbedNode => ({
  tag: 'button',
  visible: true,
  ...over,
});

const DOM: ScrubbedDom = {
  url: 'https://shop.test/cart',
  origin: 'https://shop.test',
  title: 'Cart',
  viewport: { width: 1200, height: 800, scrollX: 0, scrollY: 0 },
  nodes: [
    node({ id: 0, selector: '#accept-cookies', text: 'Accept all cookies' }),
    node({ id: 1, selector: '#email', tag: 'input', type: 'email', label: 'Email address' }),
    node({ id: 2, selector: 'button[name="checkout"]', name: 'checkout', text: 'Checkout' }),
    node({ id: 3, selector: '#disabled-btn', text: 'Pay now', disabled: true }),
  ],
  redactionSummary: {},
};

describe('extractJson', () => {
  it('reads a plain object', () => {
    expect(extractJson('{"action":"click","id":0}')).toEqual({ action: 'click', id: 0 });
  });

  it('strips ```json fences', () => {
    expect(extractJson('```json\n{"action":"done"}\n```')).toEqual({ action: 'done' });
  });

  it('ignores braces inside strings', () => {
    expect(extractJson('{"action":"fill","value":"a}b"}')).toEqual({ action: 'fill', value: 'a}b' });
  });

  it('repairs unquoted keys and single quotes', () => {
    expect(extractJson("{action: 'click', id: 2}")).toEqual({ action: 'click', id: 2 });
  });

  it('closes a truncated object', () => {
    expect(extractJson('{"action":"click","id":1')).toEqual({ action: 'click', id: 1 });
  });

  it('returns null when there is no object at all', () => {
    expect(extractJson('I am not sure what to do here.')).toBeNull();
  });
});

describe('parseAction', () => {
  it('resolves the element id to a real selector and trusts it', () => {
    const { action, confidence } = parseAction('{"action":"click","id":0}', DOM);
    expect(action).toMatchObject({ action: 'click', selector: '#accept-cookies' });
    expect(confidence).toBeGreaterThan(0.8);
  });

  it('accepts an exact selector', () => {
    const { action, confidence } = parseAction('{"action":"click","selector":"#accept-cookies"}', DOM);
    expect(action).toMatchObject({ action: 'click', selector: '#accept-cookies' });
    expect(confidence).toBeGreaterThan(0.8);
  });

  it('repairs a label used in place of a selector', () => {
    const { action, confidence } = parseAction('{"action":"click","selector":"Checkout"}', DOM);
    expect(action).toMatchObject({ action: 'click', selector: 'button[name="checkout"]' });
    expect(confidence).toBeGreaterThan(0.7);
    expect(confidence).toBeLessThan(0.8);
  });

  it('repairs a name attribute used in place of a selector', () => {
    const { action } = parseAction('{"action":"click","selector":"#checkout"}', DOM);
    expect(action).toMatchObject({ selector: 'button[name="checkout"]' });
  });

  it('maps action synonyms', () => {
    expect(parseAction('{"action":"press","id":0}', DOM).action.action).toBe('click');
    expect(parseAction('{"action":"type","id":1}', DOM).action.action).toBe('fill');
    expect(parseAction('{"action":"finish"}', DOM).action.action).toBe('done');
  });

  it('infers valueType from the field when the model omits it', () => {
    const { action } = parseAction('{"action":"fill","id":1}', DOM);
    expect(action).toMatchObject({ action: 'fill', selector: '#email', valueType: 'USER_EMAIL' });
  });

  it('distrusts a selector that is nowhere on the page', () => {
    const { action, confidence } = parseAction('{"action":"click","selector":"#totally-made-up"}', DOM);
    expect(action.action).toBe('click');
    expect(confidence).toBeLessThan(0.2);
  });

  it('penalises a disabled target', () => {
    const enabled = parseAction('{"action":"click","id":2}', DOM).confidence;
    const disabled = parseAction('{"action":"click","id":3}', DOM).confidence;
    expect(disabled).toBeLessThan(enabled);
  });

  it('escalates when there is no JSON', () => {
    const { action, confidence } = parseAction('Sorry, I cannot tell.', DOM);
    expect(action.action).toBe('escalate');
    expect(confidence).toBe(0);
  });

  it('escalates on an unrecognised action verb', () => {
    const { action, confidence } = parseAction('{"action":"teleport","id":0}', DOM);
    expect(action.action).toBe('escalate');
    expect(confidence).toBeLessThanOrEqual(0.1);
  });

  it('does not mistake the prompt schema example for an action', () => {
    // The prompt lists allowed values pipe-separated; if that ever leaks back
    // in it must not be treated as a real decision.
    const { action } = parseAction('{"action":"click|fill|scroll|done","id":0}', DOM);
    expect(action.action).toBe('escalate');
  });

  it('falls back to a viewport scroll when no element matches', () => {
    const { action } = parseAction('{"action":"scroll"}', DOM);
    expect(action).toMatchObject({ action: 'scroll', deltaY: 640 });
  });
});
