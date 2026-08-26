import { beforeEach, describe, expect, it } from 'vitest';
import { REDACTED_PLACEHOLDER } from '@shared/types';
import { buildScrubbedDom, classifyElement, cssPath } from '~/content/dom-scrubber';

const FIXTURE = `
  <h1>Checkout</h1>
  <form id="pay">
    <label for="email">Email</label>
    <input id="email" name="email" type="email" value="jane@corp.io" />
    <label for="card">Card number</label>
    <input id="card" name="cardNumber" autocomplete="cc-number" value="4242424242424242" />
    <input id="cvv" name="cvv" autocomplete="cc-csc" value="311" />
    <label for="pw">Password</label>
    <input id="pw" type="password" value="hunter2" />
    <input id="nick" name="nickname" value="janey" />
    <textarea id="notes">reach me on 555 123 4567</textarea>
    <button id="submit-btn" type="submit">Pay now</button>
  </form>
`;

describe('classifyElement', () => {
  beforeEach(() => {
    document.body.innerHTML = FIXTURE;
  });

  it('flags password inputs', () => {
    expect(classifyElement(document.getElementById('pw')!)).toContain('password');
  });

  it('flags cc-number and cc-csc via autocomplete', () => {
    expect(classifyElement(document.getElementById('card')!)).toContain('credit-card');
    expect(classifyElement(document.getElementById('cvv')!)).toContain('credit-card');
  });

  it('leaves innocuous fields unflagged', () => {
    expect(classifyElement(document.getElementById('nick')!)).toEqual([]);
  });
});

describe('cssPath', () => {
  beforeEach(() => {
    document.body.innerHTML = FIXTURE;
  });

  it('prefers a unique id and round-trips to the same element', () => {
    const el = document.getElementById('submit-btn')!;
    const selector = cssPath(el);
    expect(selector).toBe('#submit-btn');
    expect(document.querySelector(selector)).toBe(el);
  });

  it('falls back to a structural path that still resolves', () => {
    document.body.innerHTML = '<div><span>a</span><span><a href="/x">go</a></span></div>';
    const link = document.querySelector('a')!;
    const selector = cssPath(link);
    expect(document.querySelector(selector)).toBe(link);
  });
});

describe('buildScrubbedDom', () => {
  beforeEach(() => {
    document.body.innerHTML = FIXTURE;
  });

  it('masks password and card values deterministically', () => {
    const first = buildScrubbedDom(document).dom;
    const second = buildScrubbedDom(document).dom;
    expect(JSON.stringify(first.nodes)).toBe(JSON.stringify(second.nodes));

    const byId = (id: string) => first.nodes.find((n) => n.selector === `#${id}`);
    expect(byId('pw')!.value).toBe(REDACTED_PLACEHOLDER);
    expect(byId('card')!.value).toBe(REDACTED_PLACEHOLDER);
    expect(byId('cvv')!.value).toBe(REDACTED_PLACEHOLDER);
  });

  it('never emits a raw card number or password anywhere in the payload', () => {
    const serialized = JSON.stringify(buildScrubbedDom(document).dom);
    expect(serialized).not.toContain('4242424242424242');
    expect(serialized).not.toContain('hunter2');
    expect(serialized).not.toContain('jane@corp.io');
  });

  it('redacts PII found in free text', () => {
    const dom = buildScrubbedDom(document).dom;
    const notes = dom.nodes.find((n) => n.selector === '#notes');
    expect(notes?.value).toBe(REDACTED_PLACEHOLDER);
  });

  it('keeps benign values and labels for the model to reason about', () => {
    const dom = buildScrubbedDom(document).dom;
    const nick = dom.nodes.find((n) => n.selector === '#nick');
    expect(nick?.value).toBe('janey');
    expect(dom.nodes.find((n) => n.selector === '#submit-btn')?.text).toBe('Pay now');
  });

  it('counts redactions for the privacy receipt', () => {
    const { dom } = buildScrubbedDom(document);
    expect(dom.redactionSummary.password).toBeGreaterThanOrEqual(1);
    expect(dom.redactionSummary['credit-card']).toBeGreaterThanOrEqual(2);
  });

  it('strips query strings from the reported URL', () => {
    const { dom } = buildScrubbedDom(document);
    expect(dom.url).not.toContain('?');
  });
});
