import { describe, expect, it } from 'vitest';
import { decomposeGoal, decomposeWithRules } from '~/ai/nano-query-planner';

describe('nano-query-planner', () => {
  it('decomposes sequential goals separated by "and then"', () => {
    const goal = 'Search for iPhone 15 and then click the first result and then click Add to Cart';
    const subObjectives = decomposeWithRules(goal);

    expect(subObjectives.length).toBe(3);
    expect(subObjectives[0].description).toBe('Search for iPhone 15');
    expect(subObjectives[0].status).toBe('active');
    expect(subObjectives[1].description).toBe('Click the first result');
    expect(subObjectives[1].status).toBe('pending');
    expect(subObjectives[2].description).toBe('Click Add to Cart');
    expect(subObjectives[2].status).toBe('pending');
  });

  it('decomposes sequential goals separated by "then" and semicolons', () => {
    const goal = 'Go to settings; then update profile picture -> click save';
    const subObjectives = decomposeWithRules(goal);

    expect(subObjectives.length).toBe(3);
    expect(subObjectives[0].description).toBe('Go to settings');
    expect(subObjectives[1].description).toBe('Update profile picture');
    expect(subObjectives[2].description).toBe('Click save');
  });

  it('decomposes compound goals joined by "and" with action verbs', () => {
    const goal = 'Search for running shoes and add to cart';
    const subObjectives = decomposeWithRules(goal);

    expect(subObjectives.length).toBe(2);
    expect(subObjectives[0].description).toBe('Search for running shoes');
    expect(subObjectives[1].description).toBe('Add to cart');
  });

  it('keeps simple single-step goals as single sub-objective', () => {
    const goal = 'Click the login button';
    const subObjectives = decomposeWithRules(goal);

    expect(subObjectives.length).toBe(1);
    expect(subObjectives[0].description).toBe('Click the login button');
    expect(subObjectives[0].status).toBe('active');
  });

  it('decomposeGoal falls back cleanly when Chrome AI is not present', async () => {
    const res = await decomposeGoal('Search for noise cancelling headphones and add to cart');
    expect(res.subObjectives.length).toBe(2);
    expect(res.source).toBe('local-rules');
  });
});
