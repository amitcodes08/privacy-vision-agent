import { StrictMode, useCallback, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { AlertCircle, CheckCircle2, Mic, ShieldCheck } from 'lucide-react';
import './permission.css';

type PermissionStatusState = 'prompt' | 'requesting' | 'granted' | 'denied';

function PermissionApp() {
  const [state, setState] = useState<PermissionStatusState>('prompt');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const requestPermission = useCallback(async () => {
    setState('requesting');
    setErrorMessage(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((t) => t.stop());
      setState('granted');
      setTimeout(() => {
        window.close();
      }, 2000);
    } catch (err) {
      setState('denied');
      setErrorMessage(
        err instanceof Error && err.name === 'NotAllowedError'
          ? 'Microphone access was denied. Please allow microphone permissions in your browser address bar settings, then try again.'
          : 'Unable to access microphone. Please check your system audio permissions.',
      );
    }
  }, []);

  useEffect(() => {
    if (navigator.permissions?.query) {
      navigator.permissions
        .query({ name: 'microphone' as PermissionName })
        .then((res) => {
          if (res.state === 'granted') {
            setState('granted');
          } else if (res.state === 'denied') {
            setState('denied');
            setErrorMessage('Microphone access is currently blocked for this extension in browser settings.');
          } else {
            void requestPermission();
          }
        })
        .catch(() => {
          void requestPermission();
        });
    } else {
      void requestPermission();
    }
  }, [requestPermission]);

  return (
    <div className="card">
      <div className={`icon-badge ${state}`}>
        {state === 'granted' ? (
          <CheckCircle2 size={26} />
        ) : state === 'denied' ? (
          <AlertCircle size={26} />
        ) : (
          <Mic size={26} />
        )}
      </div>

      <h1>
        {state === 'granted'
          ? 'Microphone Enabled'
          : state === 'denied'
          ? 'Microphone Permission Needed'
          : 'Enable Microphone Access'}
      </h1>

      <p className="desc">
        {state === 'granted'
          ? 'Voice dictation is ready. You can now close this tab and return to Privacy Vision Agent.'
          : 'Allow microphone access to dictate queries directly into the popup input box.'}
      </p>

      <div className="privacy-notice">
        <ShieldCheck size={18} />
        <span>Audio is processed 100% on-device using local Whisper tiny. Your voice never leaves your device.</span>
      </div>

      {state === 'denied' && errorMessage && <div className="alert-box">{errorMessage}</div>}

      {state === 'granted' && (
        <div className="success-box">
          Permission granted! You can now use the microphone in the extension popup.
        </div>
      )}

      {state === 'granted' ? (
        <button className="primary" onClick={() => window.close()}>
          Close Tab
        </button>
      ) : (
        <button
          className="primary"
          onClick={() => void requestPermission()}
          disabled={state === 'requesting'}
        >
          {state === 'requesting' ? 'Requesting Permission…' : 'Allow Microphone'}
        </button>
      )}
    </div>
  );
}

const rootEl = document.getElementById('root');
if (rootEl) {
  createRoot(rootEl).render(
    <StrictMode>
      <PermissionApp />
    </StrictMode>,
  );
}
