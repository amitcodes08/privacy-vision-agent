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

/**
 * The node budget used to be spent in raw document order, so on a real
 * application the nav chrome ate it and the element the user asked about never
 * reached the model at all. jsdom reports all-zero rects, so these assert the
 * operability and accessible-name parts of the priority; the viewport term is
 * not observable here.
 */
describe('buildScrubbedDom node budget', () => {
  const CHROME_THEN_FILES = `
    <nav>
      <div role="presentation"></div>
      <div role="presentation"></div>
      <div role="none"></div>
      <div role="separator"></div>
      <div role="presentation"></div>
    </nav>
    <ul id="files">
      <li><a id="pkg" href="/repo/blob/main/package.json">package.json</a></li>
      <li><a id="readme" href="/repo/blob/main/README.md">README.md</a></li>
    </ul>
  `;

  beforeEach(() => {
    document.body.innerHTML = CHROME_THEN_FILES;
  });

  it('spends the budget on operable named elements, not on earlier bare roles', () => {
    const { dom } = buildScrubbedDom(document, 3);
    const selectors = dom.nodes.map((n) => n.selector);
    expect(selectors).toContain('#pkg');
    expect(selectors).toContain('#readme');
    expect(dom.nodes).toHaveLength(3);
  });

  it('emits the selected nodes in document order with monotonic ids', () => {
    const { dom } = buildScrubbedDom(document, 3);
    expect(dom.nodes.map((n) => n.id)).toEqual([0, 1, 2]);
    const at = (sel: string) => dom.nodes.findIndex((n) => n.selector === sel);
    expect(at('#pkg')).toBeLessThan(at('#readme'));
  });

  it('prefers the element with an accessible name when the budget is one', () => {
    document.body.innerHTML = `
      <div role="button" id="mystery"></div>
      <div role="button" id="named">Download</div>
    `;
    const { dom } = buildScrubbedDom(document, 1);
    expect(dom.nodes.map((n) => n.selector)).toEqual(['#named']);
  });

  it('indexes every emitted node back to a live element', () => {
    const { dom, index } = buildScrubbedDom(document, 3);
    for (const n of dom.nodes) expect(index.get(n.selector)).toBe(document.querySelector(n.selector));
  });

  it('extracts semantic container context for identical action buttons', () => {
    document.body.innerHTML = `
      <div class="product-card" id="c1">
        <h3>iPhone 15</h3>
        <button id="b1">Add to Cart</button>
      </div>
      <div class="product-card" id="c2">
        <h3>iPhone 17 Pro</h3>
        <button id="b2">Add to Cart</button>
      </div>
    `;
    const { dom } = buildScrubbedDom(document);
    const b1 = dom.nodes.find((n) => n.selector === '#b1');
    const b2 = dom.nodes.find((n) => n.selector === '#b2');
    expect(b1?.context).toContain('iPhone 15');
    expect(b2?.context).toContain('iPhone 17 Pro');
  });
});

