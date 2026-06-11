# Chat Frontend

This is the frontend for the chat demo. It is built using Dioxus and Tailwind CSS.

## Launching the App

See the [`../backend/README.md`](../backend/README.md) for instructions on how to run the backend. Both the frontend and backend need to be running for the app to work. Startup order does not matter.

Run the following command in the root of your project to start the frontend.

```bash
dx serve
```

And go to `http://localhost:8080/chat/` in your browser. You should see the app running.

To run for a different platform, use the `--platform platform` flag. E.g.

```bash
dx serve --platform desktop
```

## Saving and Loading State

To save the chat and tool state, click the "Save" button. To load, drag the `.json` onto the chat box.

## Troubleshooting

If you see "Loading..." for a long time, it means the backend is not running or is not reachable. Check the [backend](../backend/README.md) was launched on the correct port (`-p 8079`).

## Tailwind

To modify the look and feel, use Tailwind CSS. The styles are in `input.css`
and the compiled CSS is committed at `assets/tailwind.css`, so you only need
the compiler to *change* styles — no Node/npm required:

1. Install the standalone Tailwind CLI (a self-contained binary; grab the
   v3.x release matching `tailwind.config.js`):
   https://github.com/tailwindlabs/tailwindcss/releases
2. Run it in this directory to start the watcher:

```bash
tailwindcss -i ./input.css -o ./assets/tailwind.css --watch
```
