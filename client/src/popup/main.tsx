import { createRoot } from 'react-dom/client';
import App from './App';
import './popup.css';

const host = document.getElementById('root');
if (host) createRoot(host).render(<App />);
