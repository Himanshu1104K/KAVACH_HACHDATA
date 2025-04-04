/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'black-primary': '#121212',
        'black-secondary': '#1E1E1E',
        'gray-dark': '#2D2D2D',
        'gray-medium': '#505050',
        'gray-light': '#909090',
        'gray-accent': '#BBBBBB',
        'gray-lightest': '#E0E0E0',
      }
    },
  },
  plugins: [],
} 