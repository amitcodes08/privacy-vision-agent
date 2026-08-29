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
import { resolveElement } from '~/agent/element-resolver';

let lastIndex = new Map<string, Element>();
/** Last observed DOM — used for elementId resolution. Refreshed on every SCRAPE. */
let lastDom: import('@shared/types').ScrubbedDom = {
  url: '',
  origin: '',
  title: '',
  viewport: { width: 0, height: 0, scrollX: 0, scrollY: 0 },
  nodes: [],
  redactionSummary: {},
};

chrome.runtime.onMessage.addListener((msg: { kind?: string; action?: AgentAction; actions?: AgentAction[]; timeoutMs?: number }, _sender, sendResponse) => {
  void (async () => {
    try {
      if (msg?.kind === 'PING') {
        sendResponse({ ok: true });
        return;
      }
      if (msg?.kind === 'SCRAPE') {
        const { dom, sensitiveBoxes, index } = buildScrubbedDom(document);
        lastIndex = index;
        lastDom = dom;
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
      if (msg?.kind === 'EXECUTE_BATCH' && msg.actions && msg.actions.length > 0) {
        const details: string[] = [];
        for (const act of msg.actions) {
          const detail = await perform(act);
          details.push(detail);
        }
        sendResponse({ ok: true, detail: details.join('; ') });
        return;
      }
      if (msg?.kind === 'WAIT_FOR_SETTLED') {
        await waitForDomSettled(msg.timeoutMs ?? 1000);
        sendResponse({ ok: true });
        return;
      }
      sendResponse({ ok: false, error: `unhandled message ${String(msg?.kind)}` });
    } catch (err) {
      sendResponse({ ok: false, error: err instanceof Error ? err.message : String(err) });
    }
  })();
  return true;
});

function waitForDomSettled(maxTimeoutMs = 1000, idleDelayMs = 80): Promise<void> {
  return new Promise((resolve) => {
    let timer: ReturnType<typeof setTimeout>;

    const done = () => {
      observer.disconnect();
      clearTimeout(timer);
      resolve();
    };

    const resetTimer = () => {
      clearTimeout(timer);
      timer = setTimeout(done, idleDelayMs);
    };

    const observer = new MutationObserver(() => {
      resetTimer();
    });

    try {
      observer.observe(document.body || document.documentElement, {
        childList: true,
        subtree: true,
        attributes: true,
        characterData: true,
      });
    } catch {
      done();
      return;
    }

    setTimeout(done, maxTimeoutMs);
    resetTimer();
  });
}

function resolve(selector: string): HTMLElement {
  const cached = lastIndex.get(selector);
  if (cached?.isConnected) return getOperableElement(cached as HTMLElement);

  try {
    const el = document.querySelector<HTMLElement>(selector);
    if (el) return getOperableElement(el);
  } catch {
    // ignore querySelector syntax issues and try ID fallback
  }

  if (selector.startsWith('#')) {
    const rawId = selector.slice(1).replace(/\\/g, '').trim();
    const byId = document.getElementById(rawId);
    if (byId) return getOperableElement(byId);
  }

  // Fallback: search for elements with matching data-pva-id or aria-label
  if (selector.startsWith('#id-')) {
    const num = selector.slice(4);
    const candidate = Array.from(lastIndex.values()).find(
      (el) => el.getAttribute('data-pva-id') === num || el.id === `id-${num}`,
    );
    if (candidate && candidate.isConnected) return getOperableElement(candidate as HTMLElement);
  }

  throw new Error(`selector not found: ${selector}`);
}

/** If element is a decorative span/icon/svg inside a button or link, target the operable container */
function getOperableElement(el: HTMLElement): HTMLElement {
  if (el.tagName === 'BUTTON' || el.tagName === 'A' || el.tagName === 'INPUT' || el.tagName === 'SELECT' || el.tagName === 'TEXTAREA' || el.getAttribute('role') === 'button') {
    return el;
  }
  const parentOperable = el.closest<HTMLElement>('button, a, [role="button"], input, select, textarea, summary');
  return parentOperable || el;
}

async function perform(action: AgentAction): Promise<string> {
  switch (action.action) {
    case 'click': {
      // Prefer elementId resolution (generic) over raw CSS selector (legacy).
      let el: HTMLElement;
      if (action.elementId !== undefined) {
        el = resolveElement(
          { elementId: action.elementId, _legacySelector: action.selector },
          lastIndex,
          lastDom,
        );
        console.debug(`[EXECUTOR] click elementId=${action.elementId} → ${el.tagName}`);
      } else {
        el = resolve(action.selector);
        console.debug(`[EXECUTOR] click selector="${action.selector}"`);
      }
      const attrs = Array.from(el.attributes).map(a => `${a.name}="${a.value}"`).join(' ');
      const accName = el.getAttribute('aria-label') || el.getAttribute('title') || (el.innerText || el.textContent || '').trim();
      console.log(`\nEXECUTOR TARGET:\nelementId=${'elementId' in action ? action.elementId : 'undefined'}\nresolved element tag=${el.tagName.toLowerCase()}\nresolved element attributes=${attrs}\nresolved accessible name=${accName}`);
      el.scrollIntoView({ block: 'center', inline: 'center', behavior: 'instant' as ScrollBehavior });
      flash(el);
      robustClick(el);
      return `clicked ${action.elementId !== undefined ? `e${action.elementId}` : action.selector}`;
    }

    case 'fill': {
      let el: HTMLElement;
      if (action.elementId !== undefined) {
        el = resolveElement(
          { elementId: action.elementId, _legacySelector: action.selector },
          lastIndex,
          lastDom,
        );
        console.debug(`[EXECUTOR] fill elementId=${action.elementId} → ${el.tagName}`);
      } else {
        el = resolve(action.selector);
        console.debug(`[EXECUTOR] fill selector="${action.selector}"`);
      }
      if (!isEditable(el)) throw new Error(`target is not editable`);
      const value = await hydrate(action);
      el.scrollIntoView({ block: 'center', inline: 'center', behavior: 'instant' as ScrollBehavior });
      flash(el);
      setNativeValue(el, value);
      if (action.submit) {
        submitFormOrEnter(el);
      }
      return `filled ${action.elementId !== undefined ? `e${action.elementId}` : action.selector} from ${action.valueType}`;
    }

    case 'type': {
      // ID-first text entry: no selector needed.
      const el = resolveElement(
        {
          elementId: action.elementId,
          label: undefined,
          text: undefined,
        },
        lastIndex,
        lastDom,
      );
      console.debug(`[EXECUTOR] type elementId=${action.elementId} value=[SAFE] → ${el.tagName}`);
      if (!isEditable(el)) throw new Error(`elementId ${action.elementId} is not editable`);
      el.scrollIntoView({ block: 'center', inline: 'center', behavior: 'instant' as ScrollBehavior });
      flash(el);
      setNativeValue(el, action.value);
      if (action.submit) {
        submitFormOrEnter(el);
      }
      return `typed into e${action.elementId}`;
    }

    case 'select': {
      const el = resolveElement(
        { elementId: action.elementId },
        lastIndex,
        lastDom,
      );
      console.debug(`[EXECUTOR] select elementId=${action.elementId} value="${action.value}"`);
      if (!(el instanceof HTMLSelectElement)) {
        throw new Error(`elementId ${action.elementId} is not a <select>`);
      }
      el.value = action.value;
      el.dispatchEvent(new Event('change', { bubbles: true }));
      return `selected "${action.value}" in e${action.elementId}`;
    }

    case 'back': {
      console.debug(`[EXECUTOR] back → window.history.back()`);
      window.history.back();
      return 'navigated back';
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
    case 'invalid':
      throw new Error(`invalid action: ${action.reason}`);
  }
}


/**
 * Dispatches a complete PointerEvent and MouseEvent cascade with exact center coordinates,
 * ensuring custom SPA framework event handlers (React, Vue, Svelte) receive authentic interactions.
 */
function robustClick(el: HTMLElement): void {
  console.log(`\nACTUAL CLICK TARGET:\ntag=${el.tagName.toLowerCase()}\ntype=${el.getAttribute('type') ?? 'undefined'}\naria-label=${el.getAttribute('aria-label') ?? 'undefined'}\ntext=${(el.innerText || el.textContent || '').trim().slice(0, 50)}`);
  const rect = el.getBoundingClientRect();
  const clientX = Math.round(rect.left + Math.max(1, rect.width / 2));
  const clientY = Math.round(rect.top + Math.max(1, rect.height / 2));

  const mouseOpts: MouseEventInit = {
    bubbles: true,
    cancelable: true,
    view: window,
    clientX,
    clientY,
    buttons: 1,
  };

  try {
    el.dispatchEvent(new PointerEvent('pointerover', mouseOpts));
    el.dispatchEvent(new PointerEvent('pointerenter', mouseOpts));
    el.dispatchEvent(new PointerEvent('pointerdown', mouseOpts));
    el.dispatchEvent(new MouseEvent('mousedown', mouseOpts));
  } catch {
    // PointerEvent fallback
  }

  el.focus?.();

  try {
    el.dispatchEvent(new PointerEvent('pointerup', mouseOpts));
    el.dispatchEvent(new MouseEvent('mouseup', mouseOpts));
    el.dispatchEvent(new MouseEvent('click', mouseOpts));
  } catch {
    // PointerEvent fallback
  }

  el.click?.();

  // If a top-level overlay was at the point, trigger it as well
  try {
    const topEl = document.elementFromPoint(clientX, clientY) as HTMLElement | null;
    if (topEl && topEl !== el && !el.contains(topEl) && !topEl.contains(el)) {
      topEl.click?.();
    }
  } catch {
    // elementFromPoint safe guard
  }
}

const isEditable = (el: HTMLElement): boolean =>
  el instanceof HTMLInputElement ||
  el instanceof HTMLTextAreaElement ||
  el.getAttribute('contenteditable') === 'true' ||
  el.getAttribute('role') === 'textbox' ||
  el.getAttribute('role') === 'searchbox';

/**
 * Triggers full input events and sets value via prototype descriptors
 * so React, Angular, and Vue controlled inputs update their internal state.
 */
function setNativeValue(el: HTMLElement, value: string): void {
  el.focus();

  try {
    el.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, cancelable: true, data: value }));
  } catch {
    // fallback
  }

  if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
    const proto = Object.getPrototypeOf(el) as object;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
    if (setter) setter.call(el, value);
    else el.value = value;

    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return;
  }

  el.textContent = value;
  el.dispatchEvent(new InputEvent('input', { bubbles: true }));
}

function submitFormOrEnter(el: HTMLElement): void {
  const form = el.closest('form');
  if (form) {
    try {
      if (typeof form.requestSubmit === 'function') {
        form.requestSubmit();
        return;
      }
    } catch {
      // requestSubmit error fallback
    }

    const submitBtn = form.querySelector<HTMLElement>('button[type="submit"], input[type="submit"]');
    if (submitBtn) {
      submitBtn.click();
      return;
    }
  }

  // Dispatch enter keydown/keyup
  const keyOpts: KeyboardEventInit = {
    key: 'Enter',
    code: 'Enter',
    keyCode: 13,
    which: 13,
    bubbles: true,
    cancelable: true,
  };
  el.dispatchEvent(new KeyboardEvent('keydown', keyOpts));
  el.dispatchEvent(new KeyboardEvent('keypress', keyOpts));
  el.dispatchEvent(new KeyboardEvent('keyup', keyOpts));
}

/** Brief outline so a human watching can see what the agent touched. */
function flash(el: HTMLElement): void {
  const previous = el.style.outline;
  el.style.outline = '2px solid #7c3aed';
  setTimeout(() => {
    el.style.outline = previous;
  }, 600);
}
