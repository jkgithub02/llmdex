import type { ModelDoc } from '../../api/model/modelDoc';
import { tableRow } from './table-row';

/**
 * The table is the same facts as a card, flattened. What these cover is that it
 * flattens without inventing: a value the backend could not produce stays
 * visibly missing in a dense grid, where a blank cell would read as zero.
 */

function doc(overrides: Partial<ModelDoc> = {}): ModelDoc {
  return { model_id: 'Qwen/Qwen3-8B', ...overrides };
}

describe('tableRow', () => {
  it('splits the identifier the way the Hub writes it', () => {
    const row = tableRow(doc({ model_id: 'nvidia/Nemotron-H-8B-Base-8K' }));

    expect(row.vendor).toBe('nvidia');
    expect(row.name).toBe('Nemotron-H-8B-Base-8K');
  });

  it('carries the derived facts a reader sorts on', () => {
    const row = tableRow(
      doc({
        checkpoints: [
          {
            repo: 'Qwen/Qwen3-8B',
            derived: {
              architecture_class: 'dense transformer',
              context_length: 40960,
              params: { total: 8190735360, is_moe: false },
              vram: { total_bytes: 23360000000, assumptions: { context: 40960 } },
            },
          },
        ],
      }),
    );

    expect(row.architecture).toBe('dense transformer');
    expect(row.tone).toBe('transformer');
    expect(row.params).toBe('8.19B');
    expect(row.context).toBe('40,960');
    expect(row.checkpoints).toBe(1);
  });

  it('keeps a withheld VRAM estimate distinguishable from a missing one', () => {
    // R2.6 - refusing to cost a hybrid is the product working, and the reason
    // has to survive into the table or the cell reads as a plain gap.
    const withheld = tableRow(
      doc({
        checkpoints: [
          {
            repo: 'x/y',
            derived: {
              vram: {
                total_bytes: null,
                unreliable_reason: 'hybrid KV math',
                assumptions: { context: 8192 },
              },
            },
          },
        ],
      }),
    );

    expect(withheld.vram).toBeNull();
    expect(withheld.vramReason).toBe('hybrid KV math');
  });

  it('reads as unclassified rather than blank when nothing was derived', () => {
    const row = tableRow(doc({ checkpoints: [{ repo: 'x/y' }] }));

    expect(row.architecture).toBe('unclassified');
    expect(row.tone).toBe('unknown');
    expect(row.params).toBeNull();
    expect(row.vram).toBeNull();
    expect(row.vramReason).toBeNull();
  });

  it('survives a document with no checkpoints at all', () => {
    const row = tableRow(doc());

    expect(row.checkpoints).toBe(0);
    expect(row.architecture).toBe('unclassified');
  });
});
