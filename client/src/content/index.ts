/**
 * Content script: the only code that touches the live page.
 *
 * Responsibilities
 *  - SCRAPE:  build the scrubbed DOM + the boxes that must be blacked out.
 *  - EXECUTE: perform one hydrated action with real user-like events.
 */
import type { AgentAction } from '@shared/types';
import { buildScrubbedDom } from './dom-scrubber';
import { dedupeBoxes } from '~/privacy/canvas-redactor';
import { hydrate } from './value-hydrator';

let lastIndex = new Map<string, Element>();

chrome.runtime.onMessage.addListener((msg: { kind?: string; action?: AgentAction }, _sender, sendResponse) => {
  void (async () => {
    try {
      if (msg?.kind === 'SCRAPE') {
        const { dom, sensitiveBoxes, index } = buildScrubbedDom(document);
        lastIndex = index;
        sendResponse({
          ok: true,
          dom,
          boxes: dedupeBoxes(sensitiveBoxes),
          dpr: window.devicePixelRatio || 1,
        });
        return;
      }
      if (msg?.kind === 'EXECUTE' && msg.action) {
        const detail = await perform(msg.action);
        sendResponse({ ok: true, detail });
        return;
      }
      sendResponse({ ok: false, error: `unhandled message ${String(msg?.kind)}` });
    } catch (err) {
      sendResponse({ ok: false, error: err instanceof Error ? err.message : String(err) });
    }
  })();
  return true;
});

function resolve(selector: string): HTMLElement {
  const cached = lastIndex.get(selector);
  if (cached?.isConnected) return cached as HTMLElement;
  const el = document.querySelector<HTMLElement>(selector);
  if (!el) throw new Error(`selector not found: ${selector}`);
  return el;
}

async function perform(action: AgentAction): Promise<string> {
  switch (action.action) {
    case 'click': {
      const el = resolve(action.selector);
      el.scrollIntoView({ block: 'center', behavior: 'instant' as ScrollBehavior });
      flash(el);
      el.focus?.();
      el.click();
      return `clicked ${action.selector}`;
    }
    case 'fill': {
      const el = resolve(action.selector);
      if (!isEditable(el)) throw new Error(`${action.selector} is not editable`);
      const value = await hydrate(action);
      el.scrollIntoView({ block: 'center', behavior: 'instant' as ScrollBehavior });
      flash(el);
      setNativeValue(el, value);
      if (action.submit) el.closest('form')?.requestSubmit?.();
      // Never log `value` — it is private context by construction.
      return `filled ${action.selector} from ${action.valueType}`;
    }
    case 'scroll': {
      if (action.selector) {
        resolve(action.selector).scrollIntoView({ block: 'center', behavior: 'smooth' });
        return `scrolled to ${action.selector}`;
      }
      window.scrollBy({ top: action.deltaY ?? window.innerHeight * 0.8, behavior: 'smooth' });
      return 'scrolled viewport';
    }
    case 'navigate': {
      const url = new URL(action.url);
      if (!/^https?:$/.test(url.protocol)) throw new Error(`refusing protocol ${url.protocol}`);
      window.location.assign(url.href);
      return `navigating to ${url.origin}${url.pathname}`;
    }
    case 'wait':
      await new Promise((r) => setTimeout(r, Math.min(action.ms, 5_000)));
      return `waited ${action.ms}ms`;
    case 'escalate':
    case 'done':
      return action.action;
  }
}

const isEditable = (el: HTMLElement): boolean =>
  el instanceof HTMLInputElement ||
  el instanceof HTMLTextAreaElement ||
  el.getAttribute('contenteditable') === 'true';

/**
 * React and friends listen for native setter calls, so a plain
 * `el.value = x` is silently ignored by controlled inputs.
 */
function setNativeValue(el: HTMLElement, value: string): void {
  if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
    const proto = Object.getPrototypeOf(el) as object;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
    el.focus();
    if (setter) setter.call(el, value);
    else el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return;
  }
  el.focus();
  el.textContent = value;
  el.dispatchEvent(new InputEvent('input', { bubbles: true }));
}

/** Brief outline so a human watching can see what the agent touched. */
function flash(el: HTMLElement): void {
  const previous = el.style.outline;
  el.style.outline = '2px solid #7c3aed';
  setTimeout(() => {
    el.style.outline = previous;
  }, 600);
}
