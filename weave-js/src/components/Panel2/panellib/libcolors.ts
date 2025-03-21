import {
  constFunction,
  isAssignableTo,
  isListLike,
  isOutputNode,
  isVoidNode,
  listObjectTypePassTags,
  Node,
  NodeOrVoidNode,
  opGetRunTag,
  opMapEach,
  opPick,
  opRunId,
  taggedValue,
  typedDict,
  voidNode,
  withNamedTag,
} from '@wandb/weave/core';
import {useMemo} from 'react';

import {usePanelContext} from '.././PanelContext';

export const useColorNode = (inputNode: Node): NodeOrVoidNode => {
  const {frame} = usePanelContext();
  return useMemo(() => {
    let rowType = inputNode.type;
    // Arbitrarily limit the number of times we unnest
    let limit = 10;
    while (isListLike(rowType) && limit > 0) {
      rowType = listObjectTypePassTags(rowType);
      limit--;
    }
    if (
      frame.runColors == null ||
      isVoidNode(frame.runColors) ||
      !isAssignableTo(rowType, taggedValue(typedDict({run: 'run'}), 'any'))
    ) {
      return voidNode();
    }

    // Don't add a runColor onto an input node from the op tag-joinObj
    // because it comes from a joinAll + groupby operation.
    // This causes a panel crash because it attempts to map a run color
    // from a type that does not have the run tag.
    //
    // There is a difference in the output type of tag-joinObj on the client (this)
    // and the query engine service.
    if (isOutputNode(inputNode) && inputNode.fromOp.name === 'tag-joinObj') {
      return voidNode();
    }

    return opMapEach({
      obj: inputNode,
      mapFn: constFunction({row: withNamedTag('run', 'run', 'any')}, ({row}) =>
        opPick({
          obj: frame.runColors as Node, // Checked above
          key: opRunId({
            run: opGetRunTag({obj: row as any}),
          }),
        })
      ),
    });
  }, [frame, inputNode]);
};
