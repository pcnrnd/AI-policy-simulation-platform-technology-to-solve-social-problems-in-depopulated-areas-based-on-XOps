import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import { AppStateProvider } from "./context/AppStateContext.jsx";
import { ThemeProvider } from "./context/ThemeContext.jsx";
import "./lib/chartSetup.js";
import "./styles/index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ThemeProvider>
      <AppStateProvider>
        <App />
      </AppStateProvider>
    </ThemeProvider>
  </React.StrictMode>
);
