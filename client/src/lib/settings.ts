import { DEFAULTS } from '@shared/types';
import { DEFAULT_MODEL_KEY } from '~/ai/models';

export interface Settings {
  wsUrl: string;
  confidenceThreshold: number;
  modelKey: string;
  /** Master switch: when false the agent never escalates, even on failure. */
  allowEscalation: boolean;
  redactionStyle: 'black' | 'blur' | 'pixelate';
  maxSteps: number;
  /** Local-only identity used to hydrate server value tokens. */
  profile: {
    email?: string;
    fullName?: string;
    phone?: string;
    address?: string;
  };
}

export const DEFAULT_SETTINGS: Settings = {
  wsUrl: DEFAULTS.wsUrl,
  confidenceThreshold: DEFAULTS.confidenceThreshold,
  modelKey: DEFAULT_MODEL_KEY,
  allowEscalation: true,
  redactionStyle: 'black',
  maxSteps: 12,
  profile: {},
};

const KEY = 'pva.settings';

export async function loadSettings(): Promise<Settings> {
  const raw = await chrome.storage.local.get(KEY);
  const stored = (raw[KEY] ?? {}) as Partial<Settings>;
  return { ...DEFAULT_SETTINGS, ...stored, profile: { ...DEFAULT_SETTINGS.profile, ...stored.profile } };
}

export async function saveSettings(patch: Partial<Settings>): Promise<Settings> {
  const next = { ...(await loadSettings()), ...patch };
  await chrome.storage.local.set({ [KEY]: next });
  return next;
}

export function onSettingsChanged(fn: (s: Settings) => void): void {
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'local' && changes[KEY]) fn({ ...DEFAULT_SETTINGS, ...(changes[KEY]!.newValue as Settings) });
  });
}
