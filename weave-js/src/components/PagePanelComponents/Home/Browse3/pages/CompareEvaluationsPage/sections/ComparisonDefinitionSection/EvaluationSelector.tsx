/**
 * This is a Popover that allows the user to select an evaluation.
 */
import {Popover} from '@mui/material';
import Input from '@wandb/weave/common/components/Input';
import {parseRef, parseRefMaybe, WeaveObjectRef} from '@wandb/weave/react';
import _ from 'lodash';
import React, {useEffect, useMemo, useState} from 'react';

import {MOON_300} from '../../../../../../../../common/css/color.styles';
import {Button} from '../../../../../../../Button';
import {Tailwind} from '../../../../../../../Tailwind';
import {DEFAULT_SORT_CALLS} from '../../../CallsPage/CallsTable';
import {
  DEFAULT_FILTER_CALLS,
  useCallsForQuery,
} from '../../../CallsPage/callsTableQuery';
import {useEvaluationsFilter} from '../../../CallsPage/evaluationsFilter';
import {Id} from '../../../common/Id';
import {opNiceName} from '../../../common/opNiceName';
import {CallSchema} from '../../../wfReactInterface/wfDataModelHooksInterface';
import {ModelRefLabel} from './ModelRefLabel';

type EvaluationSelectorProps = {
  entity: string;
  project: string;
  anchorEl: HTMLElement | null;
  onSelect: (callId: string) => void;
  onClose: () => void;
  excludeEvalIds?: string[];
};

export const EvaluationSelector = ({
  entity,
  project,
  anchorEl,
  onSelect,
  onClose,
  excludeEvalIds,
}: EvaluationSelectorProps) => {
  // Calls query for just evaluations
  const evaluationsFilter = useEvaluationsFilter(entity, project);
  const page = useMemo(
    () => ({
      pageSize: 100,
      page: 0,
    }),
    []
  );
  const expandedRefCols = useMemo(() => new Set<string>(), []);
  // Don't query for output here, re-queried in tsDataModelHooksEvaluationComparison.ts
  const columns = useMemo(() => ['inputs', 'display_name'], []);
  const calls = useCallsForQuery(
    entity,
    project,
    evaluationsFilter,
    DEFAULT_FILTER_CALLS,
    page,
    DEFAULT_SORT_CALLS,
    expandedRefCols,
    columns
  );

  const evalChoices = useMemo(() => {
    if (!excludeEvalIds) {
      return calls.result;
    }
    return calls.result.filter(call => !excludeEvalIds.includes(call.callId));
  }, [calls.result, excludeEvalIds]);

  const [menuOptions, setMenuOptions] = useState<CallSchema[]>(evalChoices);
  useEffect(() => {
    setMenuOptions(evalChoices);
  }, [evalChoices]);

  const onSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const search = e.target.value;
    if (search === '') {
      setMenuOptions(evalChoices);
      return;
    }

    const filteredOptions = evalChoices.filter(call => {
      if (
        (call.displayName ?? call.spanName)
          .toLowerCase()
          .includes(search.toLowerCase())
      ) {
        return true;
      }
      if (call.callId.slice(-4).includes(search)) {
        return true;
      }
      const modelRef = parseRef(call.traceCall?.inputs.model) as WeaveObjectRef;
      if (modelRef.artifactName.toLowerCase().includes(search.toLowerCase())) {
        return true;
      }
      return false;
    });

    setMenuOptions(filteredOptions);
  };

  // Popover management
  const open = Boolean(anchorEl);
  const id = open ? 'simple-popper' : undefined;

  return (
    <Popover
      id={id}
      open={open}
      anchorEl={anchorEl}
      anchorOrigin={{
        vertical: 'bottom',
        horizontal: 'left',
      }}
      transformOrigin={{
        vertical: 'top',
        horizontal: 'left',
      }}
      slotProps={{
        paper: {
          sx: {
            marginTop: '8px',
            overflow: 'visible',
            minWidth: '200px',
          },
        },
      }}
      onClose={onClose}>
      <Tailwind>
        <div className="w-full p-12">
          <Input
            type="text"
            placeholder="Search"
            icon="search"
            iconPosition="left"
            onChange={onSearchChange}
            className="w-full"
          />
          <div className="mt-12 flex max-h-[400px] flex-col gap-2 overflow-y-auto">
            {menuOptions.length === 0 && (
              <div className="text-center text-moon-600">No evaluations</div>
            )}
            {menuOptions.map(call => {
              // Check model validity, e.g. might be invalid if the user mistakenly
              // passed a non-Model/non-Op as the `model` argument to `Evaluation.evaluate`.
              const model = call.traceCall?.inputs.model;
              const isValidModel =
                _.isString(model) && parseRefMaybe(model) !== null;
              const tooltip = isValidModel
                ? undefined
                : 'This evaluation has an invalid model ref';
              return (
                <div key={call.callId} className="flex items-center gap-2">
                  <Button
                    disabled={!isValidModel}
                    variant="ghost"
                    size="small"
                    className="w-full justify-start pb-8 pt-8 text-left font-['Source_Sans_Pro'] text-base font-normal text-moon-800"
                    tooltip={tooltip}
                    onClick={() => onSelect(call.callId)}>
                    <>
                      <span className="max-w-[250px] flex-shrink flex-grow overflow-hidden text-ellipsis whitespace-nowrap">
                        {call.displayName ?? opNiceName(call.spanName)}
                      </span>
                      <span className="flex-shrink-0">
                        <Id
                          id={call.callId}
                          type="Call"
                          className="ml-0 mr-4"
                        />
                      </span>
                      {isValidModel && (
                        <>
                          <div
                            style={{
                              width: '1px',
                              height: '100%',
                              backgroundColor: MOON_300,
                            }}
                          />
                          <ModelRefLabel modelRef={model} />
                        </>
                      )}
                    </>
                  </Button>
                </div>
              );
            })}
          </div>
        </div>
      </Tailwind>
    </Popover>
  );
};
