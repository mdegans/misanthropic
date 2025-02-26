/** @type {import('tailwindcss').Config} */
module.exports = {
  mode: "all",
  content: [
    "./src/**/*.{rs,html,css}",
    "./assets/styling/*.css",
    "./dist/**/*.html",
    "./../../misanthropic/src/dioxus.rs",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
