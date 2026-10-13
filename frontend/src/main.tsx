import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { ConversationProvider } from "./state/conversation";
import { IdentityProvider } from "./state/identity";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <IdentityProvider>
      <ConversationProvider>
        <App />
      </ConversationProvider>
    </IdentityProvider>
  </StrictMode>,
);
