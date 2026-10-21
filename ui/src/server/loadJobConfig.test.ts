import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, describe, expect, it } from 'vitest';
import { applySqliteDbPath, loadJobConfigDocument } from '../../cron/loadJobConfig';

const roots: string[] = [];

const document = {
  job: 'extension',
  config: {
    name: 'sample_job',
    process: [
      {
        type: 'diffusion_trainer',
        flag: true,
        note: null,
        lr: 0.0001,
        network: { linear: 32, conv: 16 },
      },
    ],
  },
};

const dbJobConfig = JSON.stringify(document);

const sentinelDbJobConfig = JSON.stringify({
  job: 'from-db',
  config: { process: [{ type: 'from-db' }] },
});

// Copied from PyYAML 6 yaml.dump of an OrderedDict with the same contents as `document`.
const orderedDictYaml = `!!python/object/apply:collections.OrderedDict
- - - job
    - extension
  - - config
    - !!python/object/apply:collections.OrderedDict
      - - - name
          - sample_job
        - - process
          - - !!python/object/apply:collections.OrderedDict
              - - - type
                  - diffusion_trainer
                - - flag
                  - true
                - - note
                  - null
                - - lr
                  - 0.0001
                - - network
                  - !!python/object/apply:collections.OrderedDict
                    - - - linear
                        - 32
                      - - conv
                        - 16
`;

const plainYaml = `job: extension
config:
  name: sample_job
  process:
    - type: diffusion_trainer
      flag: true
      note: null
      lr: 0.0001
      network:
        linear: 32
        conv: 16
`;

const sqliteDbPath = path.join('toolkit-root', 'aitk_db.db');

function makeRoot(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'job-config-'));
  roots.push(root);
  return root;
}

function writeConfig(root: string, contents: string): void {
  fs.writeFileSync(path.join(root, 'config.yaml'), contents);
}

afterEach(() => {
  while (roots.length > 0) {
    const root = roots.pop();
    if (root) fs.rmSync(root, { recursive: true, force: true });
  }
});

describe('loadJobConfigDocument', () => {
  it('uses the database json when config.yaml is missing', () => {
    const root = makeRoot();
    expect(loadJobConfigDocument(root, dbJobConfig)).toEqual(document);
  });

  it('loads a PyYAML OrderedDict dump instead of the database', () => {
    const root = makeRoot();
    writeConfig(root, orderedDictYaml);
    expect(loadJobConfigDocument(root, sentinelDbJobConfig)).toEqual(document);
  });

  it('loads plain yaml instead of the database', () => {
    const root = makeRoot();
    writeConfig(root, plainYaml);
    expect(loadJobConfigDocument(root, sentinelDbJobConfig)).toEqual(document);
  });

  it('rejects an empty file without using the database', () => {
    const root = makeRoot();
    writeConfig(root, '');
    expect(() => loadJobConfigDocument(root, sentinelDbJobConfig)).toThrow(/Job config must be an object/);
  });

  it('rejects a yaml list without using the database', () => {
    const root = makeRoot();
    writeConfig(root, '- a\n- b\n');
    expect(() => loadJobConfigDocument(root, sentinelDbJobConfig)).toThrow(/Job config must be an object/);
  });

  it('rejects a mapping without config.process without using the database', () => {
    const root = makeRoot();
    writeConfig(root, 'job: extension\nconfig:\n  name: sample_job\n');
    expect(() => loadJobConfigDocument(root, sentinelDbJobConfig)).toThrow(/config\.process\[0\]/);
  });

  it('rejects an unknown tag without using the database', () => {
    const root = makeRoot();
    writeConfig(root, '!!python/object/apply:os.system\n- echo\n');
    expect(() => loadJobConfigDocument(root, sentinelDbJobConfig)).toThrow(/Invalid config\.yaml/);
  });

  it('rejects a directory named config.yaml without using the database', () => {
    const root = makeRoot();
    fs.mkdirSync(path.join(root, 'config.yaml'));
    expect(() => loadJobConfigDocument(root, sentinelDbJobConfig)).toThrow(/not a file/);
  });

  it('sets sqlite_db_path on the database source and the yaml source', () => {
    const root = makeRoot();
    const fromDb = loadJobConfigDocument(root, dbJobConfig);
    applySqliteDbPath(fromDb, sqliteDbPath);
    expect(fromDb.config.process[0].sqlite_db_path).toBe(sqliteDbPath);

    writeConfig(root, orderedDictYaml);
    const fromYaml = loadJobConfigDocument(root, sentinelDbJobConfig);
    applySqliteDbPath(fromYaml, sqliteDbPath);
    expect(fromYaml.config.process[0].sqlite_db_path).toBe(sqliteDbPath);
    expect(fromYaml.config.process[0].lr).toBe(0.0001);
    expect(fromYaml.config.process[0].flag).toBe(true);
    expect(fromYaml.config.process[0].note).toBeNull();
  });
});
