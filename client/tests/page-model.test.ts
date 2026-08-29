/**
 * Tests for the semantic page model.
 *
 * Uses ScrubbedDom fixtures — no browser / Chrome APIs needed.
 */
import { describe, expect, test } from 'vitest';
import { buildPageModel } from '../src/agent/page-model';
import type { ScrubbedDom } from '@shared/types';

const mockDom: ScrubbedDom = {
  url: 'https://example.com/search',
  origin: 'https://example.com',
  title: 'Example Search',
  redactionSummary: {},
  viewport: { width: 1280, height: 720, scrollX: 0, scrollY: 0 },
  nodes: [
    {
      id: 0,
      tag: 'input',
      type: 'search',
      role: 'searchbox',
      name: 'q',
      placeholder: 'Search…',
      label: 'Search',
      selector: 'input[name="q"]',
      visible: true,
      disabled: false,
    },
    {
      id: 1,
      tag: 'button',
      role: 'button',
      text: 'Search',
      selector: 'button[type="submit"]',
      visible: true,
      disabled: false,
    },
    {
      id: 2,
      tag: 'input',
      type: 'hidden',
      selector: 'input[type="hidden"]',
      visible: false,
      disabled: false,
    },
  ],
};

describe('buildPageModel', () => {
  test('builds elements for visible nodes only', () => {
    const model = buildPageModel(mockDom);
    // node id=2 is not visible — should be excluded from elements array
    expect(model.elements).toHaveLength(2);
    expect(model.elements[0]?.elementId).toBe(0);
    expect(model.elements[1]?.elementId).toBe(1);
  });

  test('maps semantic fields correctly', () => {
    const model = buildPageModel(mockDom);
    const search = model.elements[0]!;
    expect(search.tag).toBe('input');
    expect(search.type).toBe('search');
    expect(search.role).toBe('searchbox');
    expect(search.name).toBe('q');
    expect(search.placeholder).toBe('Search…');
    expect(search.label).toBe('Search');
    expect(search.visible).toBe(true);
    expect(search.enabled).toBe(true);
  });

  test('byId returns the ScrubbedNode for any id (including invisible)', () => {
    const model = buildPageModel(mockDom);
    expect(model.byId(0)?.selector).toBe('input[name="q"]');
    expect(model.byId(2)?.id).toBe(2); // invisible node is still in the map
    expect(model.byId(999)).toBeUndefined();
  });

  test('toPromptText renders all visible elements', () => {
    const model = buildPageModel(mockDom);
    const text = model.toPromptText();
    expect(text).toContain('0:');
    expect(text).toContain('1:');
    // Hidden node (id=2) should NOT appear
    expect(text).not.toContain('2:');
  });

  test('toPromptText respects budget', () => {
    const model = buildPageModel(mockDom);
    const text = model.toPromptText(1);
    expect(text).toContain('0:');
    expect(text).not.toContain('1:');
  });

  test('toPromptText contains semantic attributes but not CSS selectors', () => {
    const model = buildPageModel(mockDom);
    const text = model.toPromptText();
    // Should have role and label
    expect(text).toContain('role=searchbox');
    expect(text).toContain('label="Search"');
    // Should NOT expose raw CSS selector
    expect(text).not.toContain('input[name');
    expect(text).not.toContain('button[type');
  });

  test('disabled node renders as disabled', () => {
    const dom: ScrubbedDom = {
      ...mockDom,
      nodes: [
        { id: 0, tag: 'button', text: 'Submit', selector: 'button', visible: true, disabled: true },
      ],
    };
    const model = buildPageModel(dom);
    const text = model.toPromptText();
    expect(text).toContain('disabled');
  });
});

describe('Page model — generic website simulation', () => {
  test('Search box on unfamiliar page identified by role/type', () => {
    // Simulate a page with no id/name attributes — pure semantic detection
    const genericDom: ScrubbedDom = {
      url: 'https://unfamiliar-site.xyz/',
      origin: 'https://unfamiliar-site.xyz',
      title: 'Welcome',
      redactionSummary: {},
      viewport: { width: 1280, height: 800, scrollX: 0, scrollY: 0 },
      nodes: [
        {
          id: 0,
          tag: 'input',
          type: 'text',
          role: 'searchbox',
          placeholder: 'What are you looking for?',
          selector: 'div > input:nth-of-type(1)',
          visible: true,
          disabled: false,
        },
        {
          id: 1,
          tag: 'button',
          role: 'button',
          text: 'Go',
          selector: 'div > button:nth-of-type(1)',
          visible: true,
          disabled: false,
        },
      ],
    };

    const model = buildPageModel(genericDom);
    // The agent should be able to find the search box by role
    const searchEl = model.elements.find((e) => e.role === 'searchbox');
    expect(searchEl).toBeDefined();
    expect(searchEl?.elementId).toBe(0);

    // The agent should be able to find the button by tag/text
    const btn = model.elements.find((e) => e.tag === 'button');
    expect(btn?.text).toBe('Go');
  });
});
