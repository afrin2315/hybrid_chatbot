/** Tailwind build config — scans the templates and emits only the utilities
 *  actually used, so we ship a small static stylesheet instead of the CDN. */
module.exports = {
  content: ["./templates/**/*.html"],
  theme: { extend: {} },
  plugins: [],
};
