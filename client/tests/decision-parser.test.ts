import { describe, expect, it } from 'vitest';
import type { ScrubbedDom, ScrubbedNode } from '@shared/types';
import { buildPrompt, extractJson, parseAction } from '~/ai/decision-parser';

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

describe('parseAction corroboration', () => {
  it('raises confidence when the ranker would have picked the same element', () => {
    const bare = parseAction('{"action":"click","id":0}', DOM).confidence;
    const vouched = parseAction('{"action":"click","id":0}', DOM, { goal: 'accept cookies' }).confidence;
    expect(vouched).toBeGreaterThan(bare);
  });

  it('does not vouch for an element unrelated to the goal', () => {
    const bare = parseAction('{"action":"click","id":2}', DOM).confidence;
    const same = parseAction('{"action":"click","id":2}', DOM, { goal: 'accept cookies' }).confidence;
    expect(same).toBe(bare);
  });

  it('still refuses an unresolvable element however relevant the goal', () => {
    const { confidence } = parseAction('{"action":"click","selector":"#nope"}', DOM, { goal: 'accept cookies' });
    expect(confidence).toBeLessThan(0.2);
  });
});

describe('buildPrompt element selection', () => {
  /** 60 filler buttons, then the one element the goal is actually about. */
  const big: ScrubbedDom = {
    ...DOM,
    nodes: [
      ...Array.from({ length: 60 }, (_, i) => node({ id: i, selector: `#filler-${i}`, text: `Item ${i}` })),
      node({ id: 60, selector: '#accept-cookies', text: 'Accept all cookies' }),
    ],
  };

  it('includes a goal-relevant element that sits far past a naive slice', () => {
    const prompt = buildPrompt('accept cookies', big);
    expect(prompt).toContain('60:');
    expect(prompt).toContain('Accept all cookies');
  });

  it('marks relevant elements so the model can find them', () => {
    expect(buildPrompt('accept cookies', big)).toMatch(/60:.*mentions your goal/);
  });

  it('stays within the element budget on a long page', () => {
    const listed = buildPrompt('accept cookies', big)
      .split('\n')
      .filter((l) => /^\d+: /.test(l));
    expect(listed.length).toBeLessThanOrEqual(36);
  });

  it('lists elements in DOM order so the list matches the screenshot', () => {
    const ids = buildPrompt('accept cookies', big)
      .split('\n')
      .filter((l) => /^\d+: /.test(l))
      .map((l) => Number(l.split(':')[0]));
    expect(ids).toEqual([...ids].sort((a, b) => a - b));
  });

  it('tells the model not to repeat what it already did', () => {
    const prompt = buildPrompt('accept cookies', big, [{ action: 'click', selector: '#filler-1' }]);
    expect(prompt).toContain('do not repeat');
  });

  it('omits disabled elements entirely', () => {
    expect(buildPrompt('pay now', DOM)).not.toContain('#disabled-btn');
    expect(buildPrompt('pay now', DOM)).not.toMatch(/^3: /m);
  });
});
