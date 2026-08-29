import { describe, expect, test } from 'vitest';
import { SpeculativePlanner } from '../src/ai/speculative-planner';
import type { ScrubbedDom, TaskMemory } from '@shared/types';

const mockDom: ScrubbedDom = {
  url: 'https://store.example.com/checkout',
  origin: 'https://store.example.com',
  title: 'Checkout Page',
  redactionSummary: {},
  viewport: {
  width: 1280,
  height: 720,
  scrollX: 0,
  scrollY: 0,
},
  nodes: [
    {
      id: 1,
      tag: 'input',
      type: 'text',
      role: 'textbox',
      selector: '#shipping-name',
      label: 'Full Name',
      text: '',
      visible: true,
      disabled: false,
    },
    {
      id: 2,
      tag: 'button',
      role: 'button',
      selector: '#continue-btn',
      text: 'Continue to Payment',
      visible: true,
      disabled: false,
    },
  ],
};

const taskMemory: TaskMemory = {
  goal: 'Fill shipping and click continue',
  currentObjective: 'Fill full name',
  attemptedTargets: [],
  completedObjectives: [],
  step: 1,
  subObjectives: [
    { id: 1, description: 'Fill full name', status: 'active' },
    { id: 2, description: 'Click continue to payment', status: 'pending' },
  ],
};

describe('SpeculativePlanner', () => {
  test('speculatively pre-grounds the next pending sub-objective', () => {
    const planner = new SpeculativePlanner();
    const plan = planner.speculateNext(taskMemory, mockDom, []);

    expect(plan).not.toBeNull();
    expect(plan?.objective).toBe('Click continue to payment');
    expect(plan?.decision.action.action).toBe('click');
    expect((plan?.decision.action as any).selector).toBe('#continue-btn');
  });

  test('successfully consumes pre-grounded decision when objective advances', () => {
    const planner = new SpeculativePlanner();
    planner.speculateNext(taskMemory, mockDom, []);

    const decision = planner.consume('Click continue to payment', mockDom);
    expect(decision).not.toBeNull();
    expect(decision?.action.action).toBe('click');
    expect((decision?.action as any).selector).toBe('#continue-btn');

    // Should be consumed and clear
    expect(planner.consume('Click continue to payment', mockDom)).toBeNull();
  });

  test('returns null if DOM altered or objective differs', () => {
    const planner = new SpeculativePlanner();
    planner.speculateNext(taskMemory, mockDom, []);

    const alteredDom: ScrubbedDom = {
      ...mockDom,
      nodes: [
        {
          id: 99,
          tag: 'a',
          selector: '#other-link',
          text: 'Completely different DOM',
          visible: true,
          disabled: false,
        },
      ],
    };

    const mismatch = planner.consume('Click continue to payment', alteredDom);
    expect(mismatch).toBeNull();
  });
});
