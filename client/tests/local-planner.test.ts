import { describe, expect, it } from 'vitest';
import type { AgentAction, ScrubbedDom, ScrubbedNode } from '@shared/types';
import { planLocally } from '~/ai/local-planner';

const node = (over: Partial<ScrubbedNode> & { id: number; selector: string }): ScrubbedNode => ({
  tag: 'button',
  visible: true,
  ...over,
});

const dom = (nodes: ScrubbedNode[]): ScrubbedDom => ({
  url: 'https://shop.test/',
  origin: 'https://shop.test',
  title: 'Shop',
  viewport: { width: 1200, height: 800, scrollX: 0, scrollY: 0 },
  nodes,
  redactionSummary: {},
});

const CONSENT = dom([
  node({ id: 0, selector: '#accept', text: 'Accept all' }),
  node({ id: 1, selector: '#reject', text: 'Reject non-essential' }),
  node({ id: 2, selector: '#more', tag: 'a', text: 'More information' }),
]);

const LOGIN = dom([
  node({ id: 0, selector: '#email', tag: 'input', type: 'email', label: 'Email address' }),
  node({ id: 1, selector: '#pw', tag: 'input', type: 'password', label: 'Password' }),
  node({ id: 2, selector: '#submit', text: 'Sign in' }),
  node({ id: 3, selector: '#search', tag: 'input', type: 'search', placeholder: 'Search products' }),
]);

describe('planLocally', () => {
  it('maps "accept cookies" onto an Accept button via synonyms', () => {
    const d = planLocally({ goal: 'accept cookies', dom: CONSENT });
    expect(d.action).toMatchObject({ action: 'click', selector: '#accept' });
    expect(d.source).toBe('heuristic');
    expect(d.confidence).toBeGreaterThan(0.4);
  });

  it('prefers a fillable field for a fill intent', () => {
    const d = planLocally({ goal: 'enter my email address', dom: LOGIN });
    expect(d.action).toMatchObject({ action: 'fill', selector: '#email', valueType: 'USER_EMAIL' });
  });

  it('prefers a clickable element for a click intent', () => {
    const d = planLocally({ goal: 'click sign in', dom: LOGIN });
    expect(d.action).toMatchObject({ action: 'click', selector: '#submit' });
  });

  it('pulls a quoted literal out of the goal for a free-text field', () => {
    const d = planLocally({ goal: 'search for "wireless mouse"', dom: LOGIN });
    expect(d.action).toMatchObject({
      action: 'fill',
      selector: '#search',
      valueType: 'LITERAL',
      value: 'wireless mouse',
    });
  });

  it('never targets a disabled element', () => {
    const withDisabled = dom([
      node({ id: 0, selector: '#accept', text: 'Accept all', disabled: true }),
      node({ id: 1, selector: '#accept2', text: 'Accept all cookies' }),
    ]);
    const d = planLocally({ goal: 'accept cookies', dom: withDisabled });
    expect(d.action).toMatchObject({ selector: '#accept2' });
  });

  it('breaks a loop instead of clicking the same element a third time', () => {
    const history: AgentAction[] = [
      { action: 'click', selector: '#accept' },
      { action: 'click', selector: '#accept' },
    ];
    const d = planLocally({ goal: 'accept cookies', dom: CONSENT, history });
    if ('selector' in d.action) expect(d.action.selector).not.toBe('#accept');
  });

  it('honours a bare scroll goal with no matching element', () => {
    const d = planLocally({ goal: 'scroll down', dom: dom([]) });
    expect(d.action).toMatchObject({ action: 'scroll', deltaY: 640 });
  });

  it('stops rather than guessing when nothing matches', () => {
    const d = planLocally({ goal: 'download the quarterly tax report', dom: CONSENT });
    expect(d.action.action).toBe('done');
    expect(d.confidence).toBeLessThan(0.5);
  });

  it('does not exceed the heuristic confidence ceiling', () => {
    const d = planLocally({ goal: 'accept all cookies consent', dom: CONSENT });
    expect(d.confidence).toBeLessThanOrEqual(0.78);
  });

  it('does whole-word matching, so "log" does not fire on "blog"', () => {
    const blog = dom([node({ id: 0, selector: '#blog', tag: 'a', text: 'Blog' })]);
    const d = planLocally({ goal: 'log in to my account', dom: blog });
    expect(d.action.action).toBe('done');
  });
});
