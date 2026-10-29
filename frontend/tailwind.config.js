/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        court: {
          50:  '#FFF7ED',
          100: '#FFEDD5',
          500: '#F97316',
          600: '#EA580C',
          900: '#7C2D12',
        },
        hoop: {
          400: '#FB923C',
          500: '#F97316',
        },
        net: {
          50:  '#F0FDF4',
          500: '#22C55E',
          700: '#15803D',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'score-in':  'scoreIn 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) both',
        'slide-up':  'slideUp 0.4s ease-out both',
        'pulse-ring':'pulseRing 1.5s ease-out infinite',
        'ball-arc':  'ballArc 2s ease-in-out infinite',
      },
      keyframes: {
        scoreIn:   { '0%': { transform: 'scale(0)', opacity: '0' }, '100%': { transform: 'scale(1)', opacity: '1' } },
        slideUp:   { '0%': { transform: 'translateY(16px)', opacity: '0' }, '100%': { transform: 'translateY(0)', opacity: '1' } },
        pulseRing: { '0%': { transform: 'scale(1)', opacity: '0.8' }, '100%': { transform: 'scale(1.8)', opacity: '0' } },
        ballArc:   { '0%,100%': { transform: 'translateX(0) translateY(0)' }, '50%': { transform: 'translateX(60px) translateY(-40px)' } },
      },
    },
  },
  plugins: [],
}
