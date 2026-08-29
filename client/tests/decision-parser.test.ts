import { describe, expect, it } from 'vitest';
import type { AgentDecision, ScrubbedDom, ScrubbedNode } from '@shared/types';
import { buildPrompt, extractJson, parseAction, parseDoneFromText, sanitiseCloudAction } from '~/ai/decision-parser';

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
    // "type" now maps to TypeAction (id-first, value required). Without value → invalid.
    expect(parseAction('{"action":"type","id":1}', DOM).action.action).toBe('invalid');
    // With value, "type" produces TypeAction
    expect(parseAction('{"action":"type","id":1,"value":"hello"}', DOM).action.action).toBe('type');
    expect(parseAction('{"action":"finish"}', DOM).action.action).toBe('done');
  });

  it('infers valueType from the field when the model omits it', () => {
    const { action } = parseAction('{"action":"fill","id":1}', DOM);
    expect(action).toMatchObject({ action: 'fill', selector: '#email', valueType: 'USER_EMAIL' });
  });

  it('distrusts a selector that is nowhere on the page', () => {
    const { action, confidence } = parseAction('{"action":"click","selector":"#made-up"}', DOM);
    expect(action.action).toBe('invalid');
    expect(confidence).toBe(0); // Invalid actions get 0 confidence
  });

  it('penalises a disabled target', () => {
    const enabled = parseAction('{"action":"click","id":2}', DOM).confidence;
    const disabled = parseAction('{"action":"click","id":3}', DOM).confidence;
    expect(disabled).toBeLessThan(enabled);
  });

  it('escalates when there is no JSON', () => {
    const { action, confidence } = parseAction('Sorry, I cannot tell.', DOM);
    expect(action.action).toBe('invalid');
    expect(confidence).toBe(0);
  });

  it('escalates on an unrecognised action verb', () => {
    const { action, confidence } = parseAction('{"action":"teleport","target":"moon"}', DOM);
    expect(action.action).toBe('invalid');
    expect(confidence).toBeLessThanOrEqual(0.1);
  });

  it('does not mistake the prompt schema example for an action', () => {
    // The prompt includes literal string "click|fill|scroll|done". A model parroting it
    // in it must not be treated as a real decision.
    const { action } = parseAction('{"action":"click|fill|scroll|done","selector":"#foo"}', DOM);
    expect(action.action).toBe('invalid');
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
    const { action, confidence } = parseAction('{"action":"click","selector":"#missing"}', DOM, { goal: 'missing' });
    expect(action.action).toBe('invalid');
    expect(confidence).toBe(0);
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

  it('returns valid when click targets a button', () => {
    const { action } = parseAction('{"action": "click", "selector": "#disabled-btn"}', DOM);
    expect(action.action).toBe('click');
  });

  it('rejects fill targeting a non-input element', () => {
    // The #disabled-btn is a button, not an input.
    const { action } = parseAction('{"action": "fill", "selector": "#disabled-btn", "value": "test"}', DOM);
    expect(action.action).toBe('invalid');
    expect('reason' in action).toBe(true);
    expect((action as any).reason).toContain('non-input');
  });

  it('rejects action with unresolvable selector', () => {
    const { action } = parseAction('{"action": "click", "selector": "#does-not-exist"}', DOM);
    expect(action.action).toBe('invalid');
    expect('reason' in action).toBe(true);
    expect((action as any).reason).toContain('not found in DOM');
  });
});

/* ================================================================== *
 * Regression: VLM done handling (bug: natural-language done → escalate)
 * ================================================================== */

describe('parseAction — VLM done handling', () => {
  it('REG-1: valid JSON done with extra fields is parsed as done', () => {
    // Model returned {"action":"done","id":"1","value":"1"} — extra fields must not
    // confuse the parser into emitting escalate.
    const { action, confidence } = parseAction('{"action":"done","id":"1","value":"1"}', DOM);
    expect(action.action).toBe('done');
    expect(confidence).toBeGreaterThanOrEqual(0.65);
  });

  it('REG-2: natural-language "GOAL is already satisfied" is parsed as done', () => {
    const { action, confidence } = parseAction(
      'The GOAL is already satisfied by what you see on the current page.',
      DOM,
    );
    expect(action.action).toBe('done');
    expect(confidence).toBeGreaterThanOrEqual(0.65);
    expect(confidence).toBeLessThan(0.75); // below HIGH_CONFIDENCE; needs corroboration
  });

  it('REG-3: "The goal is already satisfied." is parsed as done', () => {
    const { action } = parseAction('The goal is already satisfied.', DOM);
    expect(action.action).toBe('done');
  });

  it('REG-4: "The task is complete." is parsed as done', () => {
    const { action } = parseAction('The task is complete.', DOM);
    expect(action.action).toBe('done');
  });

  it('REG-5: arbitrary assistant text does NOT become done', () => {
    // Sentences that mention the word "goal" but are not explicit completion
    // statements must not trigger the done path.
    const cases = [
      'Sorry, I cannot tell.',
      'The goal here is to search for items.',
      'I believe the goal might eventually be satisfied after more steps.',
      'Let me help you achieve your goal.',
    ];
    for (const text of cases) {
      const { action } = parseAction(text, DOM);
      expect(action.action).toBe('invalid');
    }
  });

  it('REG-6: "The goal is already fully satisfied" is parsed as done', () => {
    const { action } = parseAction('The goal is already fully satisfied by the current state.', DOM);
    expect(action.action).toBe('done');
  });

  it('REG-7: "The requested task has been completed" is parsed as done', () => {
    const { action } = parseAction('The requested task has been completed.', DOM);
    expect(action.action).toBe('done');
  });
});

describe('parseDoneFromText', () => {
  it('returns true for strong completion phrases', () => {
    expect(parseDoneFromText('The GOAL is already satisfied by what you see on the current page.')).toBe(true);
    expect(parseDoneFromText('The goal is satisfied.')).toBe(true);
    expect(parseDoneFromText('The task is complete.')).toBe(true);
    expect(parseDoneFromText('The goal is already fully satisfied.')).toBe(true);
    expect(parseDoneFromText('Goal is already satisfied — all items match.')).toBe(true);
  });

  it('returns false for non-completion text', () => {
    expect(parseDoneFromText('Sorry, I cannot tell.')).toBe(false);
    expect(parseDoneFromText('Let me help you achieve your goal.')).toBe(false);
    expect(parseDoneFromText('The goal here is to search for items.')).toBe(false);
    expect(parseDoneFromText('')).toBe(false);
    expect(parseDoneFromText('done')).toBe(false); // too short / not a completion statement
  });

  it('does NOT match goal-as-noun in the middle of a sentence', () => {
    // "goal" appearing in a subordinate clause must not fire.
    expect(parseDoneFromText('I think the goal might be satisfied after more steps.')).toBe(false);
  });
});

describe('sanitiseCloudAction — cloud decision validation', () => {
  it('REG-8: rejects a cloud click action that contains fill-specific fields', () => {
    const dirty: AgentDecision = {
      action: {
        action: 'click',
        selector: '#root > header > input',
        // @ts-expect-error — intentionally malformed cloud response
        valueType: 'USER_EMAIL',
        value: 'Test Privacy Vision Agent',
      },
      confidence: 0.8,
      source: 'cloud',
    };
    const clean = sanitiseCloudAction(dirty);
    expect(clean.action.action).toBe('invalid');
    expect('reason' in clean.action).toBe(true);
  });

  it('leaves a well-formed click action unchanged', () => {
    const good: AgentDecision = {
      action: { action: 'click', selector: '#btn', reason: 'click the button' },
      confidence: 0.9,
      source: 'cloud',
    };
    expect(sanitiseCloudAction(good)).toBe(good); // same reference → no copy made
  });

  it('leaves non-click actions unchanged', () => {
    const fill: AgentDecision = {
      action: { action: 'fill', selector: '#email', valueType: 'USER_EMAIL' },
      confidence: 0.9,
      source: 'cloud',
    };
    expect(sanitiseCloudAction(fill)).toBe(fill);

    const done: AgentDecision = {
      action: { action: 'done', summary: 'task complete' },
      confidence: 1,
      source: 'cloud',
    };
    expect(sanitiseCloudAction(done)).toBe(done);
  });
});
