/**
 * Resolves server-issued value tokens against local private context.
 * The server never sees these strings — it only names them.
 */
import type { FillAction, ValueToken } from '@shared/types';
import { loadSettings } from '~/lib/settings';

/** Tokens we refuse to hydrate from a cloud instruction, full stop. */
const FORBIDDEN: ReadonlySet<ValueToken> = new Set(['USER_PASSWORD', 'OTP_CODE']);

export class HydrationError extends Error {}

export async function hydrate(action: FillAction): Promise<string> {
  if (FORBIDDEN.has(action.valueType)) {
    // A remote model must not be able to make the extension type a
    // credential. Credentials stay a manual, human action.
    throw new HydrationError(`refusing to hydrate ${action.valueType} from a remote instruction`);
  }
  if (action.valueType === 'LITERAL') {
    if (typeof action.value !== 'string') throw new HydrationError('LITERAL fill missing value');
    return action.value;
  }
  const { profile } = await loadSettings();
  const map: Partial<Record<ValueToken, string | undefined>> = {
    USER_EMAIL: profile.email,
    USER_FULL_NAME: profile.fullName,
    USER_PHONE: profile.phone,
    USER_ADDRESS: profile.address,
  };
  const value = map[action.valueType];
  if (!value) throw new HydrationError(`no local value stored for ${action.valueType}`);
  return value;
}
