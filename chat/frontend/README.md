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

To modify the look and feel, use Tailwind CSS. The styles are in `input.css` and the compiled CSS is in `assets/tailwind.css`. To compile the CSS, you need to have Node.js and npm installed, sorry.

1. Install npm: https://docs.npmjs.com/downloading-and-installing-node-js-and-npm
2. Install the Tailwind CSS CLI: https://tailwindcss.com/docs/installation
3. Run the following command in the root of the project to start the Tailwind CSS compiler:

```bash
npx tailwindcss -i ./input.css -o ./assets/tailwind.css --watch
```
