/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // V7P3R Viper Theme
        'viper': {
          'black': '#000000',
          'dark': '#0a0f0a',
          'green': '#00ff41',
          'green-dim': '#00aa2a',
          'green-dark': '#005515',
          'text-muted': '#6b8f71',
        },
      },
      fontFamily: {
        'mono': ['"Space Mono"', 'monospace'],
        'sans': ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
