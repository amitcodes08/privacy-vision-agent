import { describe, expect, test } from 'vitest';
import { ActionCache } from '../src/ai/action-cache';
import type { ScrubbedDom } from '@shared/types';

const mockDom: ScrubbedDom = {
  url: 'https://example.com/search',
  origin: 'https://example.com',
  title: 'Test Page',
  viewport: { width: 1280, height: 720, scrollX: 0, scrollY: 0 },
  redactionSummary: {},
  nodes: [
    {
      id: 1,
      tag: 'input',
      type: 'text',
      role: 'searchbox',
      selector: '#search-box',
      text: '',
      label: 'Search input',
      visible: true,
      disabled: false,
      box: { x: 10, y: 10, width: 100, height: 30 },
    },
    {
      id: 2,
      tag: 'button',
      role: 'button',
      selector: '#search-btn',
      text: 'Search',
      visible: true,
      disabled: false,
      box: { x: 120, y: 10, width: 60, height: 30 },
    },
  ],
};

describe('ActionCache', () => {
  test('generates consistent structural fingerprint', () => {
    const cache = new ActionCache();
    const fp1 = cache.computeFingerprint(mockDom);
    const fp2 = cache.computeFingerprint(mockDom);
    expect(fp1).toBe(fp2);
    expect(fp1.length).toBeGreaterThan(0);
  });

  test('stores and retrieves cached decision on DOM match', () => {
    const cache = new ActionCache();
    const origin = 'https://example.com';
    const obj = 'Search for shoes';
    const actions = [
      { action: 'fill' as const, selector: '#search-box', valueType: 'LITERAL' as const, value: 'shoes', submit: true },
    ];

    cache.set(origin, obj, mockDom, actions, 0.95);

    const hit = cache.get(origin, obj, mockDom);
    expect(hit).not.toBeNull();
    expect(hit?.action.action).toBe('fill');
    expect(hit?.source).toBe('cache');
    expect(cache.getHits()).toBe(1);
  });

  test('returns null when DOM fingerprint differs', () => {
    const cache = new ActionCache();
    const origin = 'https://example.com';
    const obj = 'Search for shoes';
    const actions = [
      { action: 'click' as const, selector: '#search-btn' },
    ];

    cache.set(origin, obj, mockDom, actions, 0.95);

    const alteredDom: ScrubbedDom = {
      ...mockDom,
      nodes: [
        {
          id: 99,
          tag: 'a',
          role: 'link',
          selector: '#different-link',
          text: 'Completely different DOM',
          visible: true,
          disabled: false,
        },
      ],
    };

    const miss = cache.get(origin, obj, alteredDom);
    expect(miss).toBeNull();
  });

  test('invalidates cache correctly', () => {
    const cache = new ActionCache();
    const origin = 'https://example.com';
    const obj = 'Search for shoes';

    cache.set(origin, obj, mockDom, [{ action: 'click', selector: '#search-btn' }], 0.95);
    cache.invalidate(origin, obj);

    expect(cache.get(origin, obj, mockDom)).toBeNull();
  });
});
