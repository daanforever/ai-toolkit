import type { DocumentOptions, ScalarTag, SchemaOptions, Tags } from 'yaml';

// PyYAML's decimal int is `0` or a non-zero start. `048` is not octal (digit 8)
// and not decimal, so it stays a string. The yaml 1.1 schema otherwise accepts
// any digit run, which turns `048` into 48.
const PYYAML_DECIMAL_INT = /^[-+]?(?:0|[1-9][0-9_]*)$/;

function isPyyamlDecimalIntTag(tag: Tags[number]): tag is ScalarTag {
  return typeof tag !== 'string' && tag.tag === 'tag:yaml.org,2002:int' && tag.format == null && tag.test != null;
}

export function withPyyamlDecimalInt(tags: Tags): Tags {
  return tags.map(tag => (isPyyamlDecimalIntTag(tag) ? { ...tag, test: PYYAML_DECIMAL_INT } : tag));
}

export const PYYAML_COMPATIBLE_OPTIONS: DocumentOptions & SchemaOptions = {
  version: '1.1',
  customTags: withPyyamlDecimalInt,
};

// `compat: 'core'` quotes scalars that YAML 1.2 would read as a non-string (`048`).
// Parsing with `compat` emits TAG_RESOLVE_FAILED warnings, so parse paths use
// PYYAML_COMPATIBLE_OPTIONS and only stringify uses this.
export const PYYAML_STRINGIFY_OPTIONS: DocumentOptions & SchemaOptions = {
  ...PYYAML_COMPATIBLE_OPTIONS,
  compat: 'core',
};
