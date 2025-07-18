/**
 * Each filter in the filter popup is shown as one row.
 */

import {GridFilterItem} from '@mui/x-data-grid-pro';
import React, {useMemo} from 'react';

import {Button} from '../../../../Button';
import {
  FilterId,
  getFieldType,
  getGroupedOperatorOptions,
  isWeaveRef,
} from './common';
import {SelectField, SelectFieldOption} from './SelectField';
import {SelectOperator} from './SelectOperator';
import {SelectValue} from './SelectValue';

type FilterRowProps = {
  entity: string;
  project: string;
  item: GridFilterItem;
  options: SelectFieldOption[];
  onAddFilter: (field: string) => void;
  onUpdateFilter: (item: GridFilterItem) => void;
  onRemoveFilter: (id: FilterId) => void;
  activeEditId?: FilterId | null;
  disabled?: boolean;
};

export const FilterRow = ({
  entity,
  project,
  item,
  options,
  onAddFilter,
  onUpdateFilter,
  onRemoveFilter,
  activeEditId,
  disabled = false,
}: FilterRowProps) => {
  const onSelectField = (field: string) => {
    if (item.id == null) {
      onAddFilter(field);
    } else {
      // If this is an additional filter, we need to get new operator
      // because the default is string
      const newOperatorOptions = getGroupedOperatorOptions(field);
      const operator = newOperatorOptions[0].options[0].value;

      // Check if field type changed (eg str -> int) and clear value if so
      const oldFieldType = getFieldType(item.field);
      const newFieldType = getFieldType(field);
      const value = oldFieldType !== newFieldType ? undefined : item.value;

      onUpdateFilter({...item, field, operator, value});
    }
  };

  const operatorOptions = useMemo(
    () => getGroupedOperatorOptions(item.field),
    [item.field]
  );

  const onSelectOperator = (operator: string) => {
    onUpdateFilter({...item, operator});
  };

  const onSetValue = (value: string) => {
    onUpdateFilter({...item, value});
  };

  const isOperatorDisabled =
    isWeaveRef(item.value) ||
    ['id', 'status', 'run', 'user', 'monitor'].includes(
      getFieldType(item.field)
    );

  return (
    <>
      <div className="min-w-[250px]">
        <SelectField
          options={options}
          value={item.field}
          onSelectField={onSelectField}
          isDisabled={disabled}
        />
      </div>
      <div className="w-[165px]">
        {item.field && (
          <SelectOperator
            options={operatorOptions}
            value={item.operator}
            onSelectOperator={onSelectOperator}
            isDisabled={disabled || isOperatorDisabled}
          />
        )}
      </div>
      <div className="flex items-center">
        {item.field && (
          <SelectValue
            entity={entity}
            project={project}
            field={item.field}
            operator={item.operator}
            value={item.value}
            onSetValue={onSetValue}
            activeEditId={activeEditId}
            itemId={item.id}
          />
        )}
      </div>
      <div className="mb-0 flex items-center justify-center">
        {item.id != null && (
          <Button
            size="small"
            variant="ghost"
            icon="close"
            tooltip="Remove this filter"
            onClick={() => onRemoveFilter(item.id)}
          />
        )}
      </div>
    </>
  );
};
