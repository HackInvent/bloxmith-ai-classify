/** Bind the release-owned editor in the narrow inspector through the public action API. */
import { enhanceProperties } from "./properties.js";
import { bindEditor } from "./settings.js";
export function mount(root, api) {
  const disposeStyle = enhanceProperties(root), disposeEditor = bindEditor(root, api);
  return { dispose() { disposeEditor(); disposeStyle(); } };
}
