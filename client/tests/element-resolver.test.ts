/**
 * Tests for the generic element resolver.
 *
 * Uses jsdom (provided by the vitest environment) to test element lookup
 * without any website-specific assumptions.
 */
import { describe, expect, test, beforeEach } from 'vitest';
import { resolveElement, ElementNotFoundError } from '../src/agent/element-resolver';
import type { ScrubbedDom } from '@shared/types';

/* ------------------------------------------------------------------ *
 * Test fixtures
 * ------------------------------------------------------------------ */

function makeDom(): ScrubbedDom {
  return {
    url: 'https://test.example.com/',
    origin: 'https://test.example.com',
    title: 'Test Page',
    redactionSummary: {},
    viewport: { width: 1280, height: 720, scrollX: 0, scrollY: 0 },
    nodes: [
      {
        id: 0,
        tag: 'input',
        type: 'text',
        role: 'searchbox',
        placeholder: 'Search',
        label: 'Search',
        selector: '#search-box',
        visible: true,
        disabled: false,
      },
      {
        id: 1,
        tag: 'button',
        role: 'button',
        text: 'Search',
        selector: '#search-btn',
        visible: true,
        disabled: false,
      },
    ],
  };
}

function makeIndex(dom: Document): Map<string, Element> {
  const map = new Map<string, Element>();
  const searchInput = dom.getElementById('search-box');
  const searchBtn = dom.getElementById('search-btn');
  if (searchInput) map.set('#search-box', searchInput);
  if (searchBtn) map.set('#search-btn', searchBtn);
  return map;
}

/* ------------------------------------------------------------------ *
 * JSDOM setup
 * ------------------------------------------------------------------ */

beforeEach(() => {
  document.body.innerHTML = `
    <div>
      <input id="search-box" type="text" placeholder="Search" aria-label="Search" />
      <button id="search-btn">Search</button>
      <label for="email-field">Email address</label>
      <input id="email-field" type="email" name="email" />
      <select id="country-select" name="country">
        <option value="in">India</option>
        <option value="us">United States</option>
      </select>
    </div>
  `;
});

describe('resolveElement — Path 1: elementId', () => {
  test('resolves via elementId using the node index', () => {
    const dom = makeDom();
    const index = makeIndex(document);

    const el = resolveElement({ elementId: 0 }, index, dom);
    expect(el.id).toBe('search-box');
  });

  test('resolves via elementId using querySelector when not in index', () => {
    const dom = makeDom();
    // Empty index — forces querySelector path
    const el = resolveElement({ elementId: 0 }, new Map(), dom);
    expect(el.id).toBe('search-box');
  });

  test('falls through to label when elementId has no matching node in scrubbed DOM', () => {
    const dom = makeDom();
    // elementId=99 does not exist in scrubbed DOM nodes — falls to label path
    const el = resolveElement({ elementId: 99, label: 'Email address' }, new Map(), dom);
    expect((el as HTMLElement).id).toBe('email-field');
  });
});

describe('resolveElement — Path 3: label', () => {
  test('resolves by aria-label', () => {
    const el = resolveElement({ label: 'Search' }, new Map(), makeDom());
    expect(el.id).toBe('search-box');
  });

  test('resolves via associated <label> element', () => {
    const dom = makeDom();
    const el = resolveElement({ label: 'Email address' }, new Map(), dom);
    expect(el.id).toBe('email-field');
  });
});

describe('resolveElement — Path 4: visible text', () => {
  test('resolves button by text', () => {
    const el = resolveElement({ text: 'Search' }, new Map(), makeDom());
    expect(el.id).toBe('search-btn');
  });
});

describe('resolveElement — Path 5: placeholder', () => {
  test('resolves input by placeholder', () => {
    // Need CSS.escape — available in jsdom
    const el = resolveElement({ placeholder: 'Search' }, new Map(), makeDom());
    expect(el.id).toBe('search-box');
  });
});

describe('resolveElement — Path 6: name attribute', () => {
  test('resolves input by name', () => {
    const el = resolveElement({ name: 'email', type: 'email' }, new Map(), makeDom());
    expect(el.id).toBe('email-field');
  });
});

describe('resolveElement — Path 7: legacy selector', () => {
  test('resolves via legacy CSS selector', () => {
    const el = resolveElement({ _legacySelector: '#search-btn' }, new Map(), makeDom());
    expect(el.id).toBe('search-btn');
  });

  test('resolves via index when provided', () => {
    const index = makeIndex(document);
    const el = resolveElement({ _legacySelector: '#search-btn' }, index, makeDom());
    expect(el.id).toBe('search-btn');
  });
});

describe('resolveElement — errors', () => {
  test('throws ElementNotFoundError when nothing matches', () => {
    expect(() =>
      resolveElement({ elementId: 99, text: 'nonexistent-xyz' }, new Map(), makeDom()),
    ).toThrow(ElementNotFoundError);
  });
});

describe('Three functional tests (simulated)', () => {
  // Test 1: Find search box and type
  test('Test 1: agent finds search box and types', () => {
    // Simulate: agent observes page → builds DOM → identifies elementId 0 as searchbox
    const dom = makeDom();
    const index = makeIndex(document);

    // Agent action: type(elementId=0, value="laptops")
    const el = resolveElement({ elementId: 0 }, index, dom);
    expect(el.tagName).toBe('INPUT');
    expect(el.getAttribute('placeholder')).toBe('Search');
    // Executor would call setNativeValue(el, 'laptops') — not tested here
  });

  // Test 2: Click the search button
  test('Test 2: agent clicks search button', () => {
    const dom = makeDom();
    const index = makeIndex(document);

    // Agent action: click(elementId=1)
    const el = resolveElement({ elementId: 1 }, index, dom);
    expect(el.tagName).toBe('BUTTON');
    expect(el.textContent?.trim()).toBe('Search');
  });

  // Test 3: Page change invalidates old elementIds
  test('Test 3: after page change, old elementId resolves to new scrape', () => {
    const oldDom = makeDom();
    // Simulate page navigation: new DOM with different node ordering
    document.body.innerHTML = `<button id="results-back">Back to results</button>`;
    const newDom: ScrubbedDom = {
      ...oldDom,
      url: 'https://test.example.com/results',
      nodes: [
        {
          id: 0,
          tag: 'button',
          text: 'Back to results',
          selector: '#results-back',
          visible: true,
          disabled: false,
        },
      ],
    };

    // Old elementId 0 (search box) no longer valid — but new observation maps id=0 to the back button
    const el = resolveElement({ elementId: 0 }, new Map(), newDom);
    expect(el.textContent?.trim()).toBe('Back to results');
  });
});
