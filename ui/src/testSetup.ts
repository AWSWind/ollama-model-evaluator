/**
 * Test setup for Vitest (`vite.config.ts` pulls this in via ``setupFiles``).
 *
 * Adds ``@testing-library/jest-dom``'s custom matchers (e.g.
 * ``toBeInTheDocument``) to Vitest's ``expect`` so component tests
 * can assert against the DOM idiomatically.
 */
import "@testing-library/jest-dom/vitest";
