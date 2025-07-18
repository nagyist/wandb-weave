/**
 * Customized version of react-select.
 *
 * TODO: This is a first attempt to align the component UI with the design team's desired specs:
 * https://www.figma.com/file/Z6fObCiWnXEVBTtxCpKT4r/Specs?node-id=110%3A904&t=QmO1vN7DlR5knw3Z-0
 * It doesn't match the spec completely yet; waiting on UI framework decisions before investing
 * more time in alignment.
 */

import {
  hexToRGB,
  MOON_100,
  MOON_150,
  MOON_250,
  MOON_350,
  MOON_500,
  MOON_800,
  RED_550,
  TEAL_300,
  TEAL_350,
  TEAL_400,
  TEAL_600,
} from '@wandb/weave/common/css/globals.styles';
import {Icon} from '@wandb/weave/components/Icon';
import React from 'react';
import ReactSelect, {
  ClearIndicatorProps,
  components,
  DropdownIndicatorProps,
  GroupBase,
  GroupHeadingProps,
  Props,
  StylesConfig,
} from 'react-select';
import AsyncSelect, {AsyncProps} from 'react-select/async';
import AsyncCreatableSelect, {
  AsyncCreatableProps,
} from 'react-select/async-creatable';

export const SelectSizes = {
  Small: 'small',
  Medium: 'medium',
  Large: 'large',
  Variable: 'variable',
} as const;
export type SelectSize = (typeof SelectSizes)[keyof typeof SelectSizes];

const HEIGHTS: Record<SelectSize, number | undefined> = {
  small: 24,
  medium: 32,
  large: 40,
  variable: undefined,
} as const;

const MIN_HEIGHTS: Record<SelectSize, number | undefined> = {
  small: undefined,
  medium: undefined,
  large: undefined,
  variable: 32,
} as const;

const LINE_HEIGHTS: Record<SelectSize, string | undefined> = {
  small: '20px',
  medium: '24px',
  large: '24px',
  variable: undefined,
} as const;

const FONT_SIZES: Record<SelectSize, string> = {
  small: '14px',
  medium: '16px',
  large: '16px',
  variable: '14px',
} as const;

const PADDING: Record<SelectSize, string> = {
  small: '2px 8px',
  medium: '4px 12px',
  large: '8px 12px',
  variable: '4px 12px',
} as const;

const OUTWARD_MARGINS: Record<SelectSize, string> = {
  small: '-8px',
  medium: '-12px',
  large: '-12px',
  variable: '-8px',
} as const;

export type AdditionalProps = {
  size?: SelectSize;
  errorState?: boolean;
  groupDivider?: boolean;
  cursor?: string;
};

const ClearIndicator = <
  Option,
  IsMulti extends boolean = false,
  Group extends GroupBase<Option> = GroupBase<Option>
>(
  props: ClearIndicatorProps<Option, IsMulti, Group>
) => {
  return (
    <components.ClearIndicator {...props}>
      <Icon
        name="close"
        height={16}
        width={16}
        role="button"
        aria-label="Clear all"
      />
    </components.ClearIndicator>
  );
};

// Toggle icon when open
const DropdownIndicator = <
  Option,
  IsMulti extends boolean = false,
  Group extends GroupBase<Option> = GroupBase<Option>
>(
  indicatorProps: DropdownIndicatorProps<Option, IsMulti, Group>
) => {
  if (indicatorProps.isMulti) {
    return null;
  }
  const iconName = indicatorProps.selectProps.menuIsOpen
    ? 'chevron-up'
    : 'chevron-down';
  return (
    <components.DropdownIndicator {...indicatorProps}>
      <Icon name={iconName} width={16} height={16} color={MOON_500} />
    </components.DropdownIndicator>
  );
};

const getGroupHeading = <
  Option,
  IsMulti extends boolean = false,
  Group extends GroupBase<Option> = GroupBase<Option>
>(
  size: SelectSize,
  showDivider: boolean
) => {
  return (groupHeadingProps: GroupHeadingProps<Option, IsMulti, Group>) => {
    const isFirstGroup =
      groupHeadingProps.selectProps.options.findIndex(
        option => option === groupHeadingProps.data
      ) === 0;
    return (
      <components.GroupHeading {...groupHeadingProps}>
        {showDivider && !isFirstGroup && (
          <div
            style={{
              backgroundColor: MOON_250,
              height: '1px',
              margin: `0 ${OUTWARD_MARGINS[size]} 6px ${OUTWARD_MARGINS[size]}`,
            }}
          />
        )}
        {groupHeadingProps.children}
      </components.GroupHeading>
    );
  };
};

type StylesProps = {
  size?: SelectSize;
  errorState?: boolean;
  cursor?: string;
};

// Override styling to come closer to design spec.
// See: https://react-select.com/home#custom-styles
// See: https://github.com/JedWatson/react-select/issues/2728
const getStyles = <
  Option,
  IsMulti extends boolean = false,
  Group extends GroupBase<Option> = GroupBase<Option>
>(
  props: StylesProps
) => {
  const errorState = props.errorState ?? false;
  const size = props.size ?? 'medium';
  return {
    // No vertical line to left of dropdown indicator
    indicatorSeparator: baseStyles => ({...baseStyles, display: 'none'}),
    clearIndicator: baseStyles => ({
      ...baseStyles,
      padding: '4px 12px 4px 4px',
      cursor: 'pointer',
    }),
    input: baseStyles => {
      return {
        ...baseStyles,
        padding: 0,
        margin: 0,
      };
    },
    valueContainer: (baseStyles, state) => {
      const padding = state.isMulti && state.hasValue ? '4px' : PADDING[size];
      return {...baseStyles, padding, gap: '2px'};
    },
    multiValue: baseStyles => ({
      ...baseStyles,
      backgroundColor: hexToRGB(MOON_150),
      borderRadius: '3px',
      margin: '2px',
      minHeight: '22px',
      maxHeight: '22px',
      display: 'flex',
      alignItems: 'center',
    }),
    multiValueLabel: baseStyles => {
      const fontSize = FONT_SIZES[size];
      return {
        ...baseStyles,
        fontSize,
        color: MOON_800,
        paddingLeft: '6px',
        paddingRight: '6px',
        paddingTop: '0',
        paddingBottom: '0',
      };
    },
    multiValueRemove: baseStyles => ({
      ...baseStyles,
      cursor: 'pointer',
      color: MOON_500,
      paddingLeft: '1px',
      paddingRight: '6px',
      borderRadius: '0 3px 3px 0',
      backgroundColor: 'transparent',
      ':hover': {
        backgroundColor: 'transparent',
        color: MOON_800,
      },
    }),
    dropdownIndicator: baseStyles => ({...baseStyles, padding: '0 8px 0 0'}),
    container: baseStyles => {
      const height = HEIGHTS[size];
      return {
        ...baseStyles,
        height,
      };
    },
    control: (baseStyles, state) => {
      const colorBorderDefault = MOON_250;
      const colorBorderHover = TEAL_350;
      const colorBorderOpen = errorState ? hexToRGB(RED_550, 0.64) : TEAL_400;
      const height = HEIGHTS[size];
      const minHeight = MIN_HEIGHTS[size] ?? height;
      const lineHeight = LINE_HEIGHTS[size];
      const fontSize = FONT_SIZES[size];
      return {
        ...baseStyles,
        height,
        minHeight,
        lineHeight,
        fontSize,
        cursor: props.cursor ?? 'default',
        border: 0,
        borderRadius: '5px',
        outline:
          state.menuIsOpen || state.isFocused
            ? `2px solid ${colorBorderOpen}`
            : `1px solid ${colorBorderDefault}`,
        transition: 'none',
        '&:hover': {
          outline: `2px solid ${
            state.menuIsOpen || state.isFocused
              ? colorBorderOpen
              : colorBorderHover
          }`,
        },
      };
    },
    menu: baseStyles => ({
      ...baseStyles,
      // TODO: Semantic-UI based dropdowns have their z-index set to 3,
      //       which causes their selected value to appear in front of the
      //       react-select popup. We should remove this hack once we've
      //       eliminated Semantic-UI based dropdowns.
      zIndex: 4,
    }),
    menuList: baseStyles => {
      return {
        ...baseStyles,
        padding: '6px 0',
      };
    },
    groupHeading: (baseStyles, state) => {
      return {
        ...baseStyles,
        fontSize: '16px',
        fontWeight: 600,
        color: MOON_800,
        textTransform: 'none',
      };
    },
    option: (baseStyles, state) => {
      return {
        ...baseStyles,
        cursor: state.isDisabled ? 'default' : 'pointer',
        // TODO: Should icon be translucent?
        color: state.isDisabled
          ? MOON_350
          : state.isSelected
          ? TEAL_600
          : MOON_800,
        padding: '6px 10px',
        margin: '0 6px',
        borderRadius: '4px',
        width: 'auto',
        backgroundColor: state.isDisabled
          ? undefined
          : state.isSelected
          ? hexToRGB(TEAL_300, 0.32)
          : state.isFocused
          ? MOON_100
          : undefined,
        ':active': {
          // mousedown
          ...baseStyles[':active'],
          backgroundColor: !state.isDisabled
            ? hexToRGB(TEAL_300, 0.32)
            : undefined,
        },
      };
    },
    placeholder: baseStyles => ({...baseStyles, fontSize: '16px'}),
  } as StylesConfig<Option, IsMulti, Group>;
};

// See: https://react-select.com/typescript
export const Select = <
  Option,
  IsMulti extends boolean = false,
  Group extends GroupBase<Option> = GroupBase<Option>
>(
  props: Props<Option, IsMulti, Group> & AdditionalProps
) => {
  const styles: StylesConfig<Option, IsMulti, Group> = getStyles(props);
  const size = props.size ?? 'medium';
  const showDivider = props.groupDivider ?? false;
  const GroupHeading = getGroupHeading(size, showDivider);

  return (
    <ReactSelect
      {...props}
      components={Object.assign(
        {ClearIndicator, DropdownIndicator, GroupHeading},
        props.components
      )}
      styles={styles}
    />
  );
};

export const SelectAsync = <
  Option,
  IsMulti extends boolean = false,
  Group extends GroupBase<Option> = GroupBase<Option>
>(
  props: AsyncProps<Option, IsMulti, Group> & AdditionalProps
) => {
  const styles: StylesConfig<Option, IsMulti, Group> = getStyles(props);
  const size = props.size ?? 'medium';
  const showDivider = props.groupDivider ?? false;
  const GroupHeading = getGroupHeading(size, showDivider);
  return (
    <AsyncSelect
      {...props}
      components={Object.assign(
        {DropdownIndicator, GroupHeading},
        props.components
      )}
      styles={styles}
    />
  );
};

export const SelectAsyncCreatable = <
  Option,
  IsMulti extends boolean = false,
  Group extends GroupBase<Option> = GroupBase<Option>
>(
  props: AsyncCreatableProps<Option, IsMulti, Group> & AdditionalProps
) => {
  const styles: StylesConfig<Option, IsMulti, Group> = getStyles(props);
  const size = props.size ?? 'medium';
  const showDivider = props.groupDivider ?? false;
  const GroupHeading = getGroupHeading(size, showDivider);
  return (
    <AsyncCreatableSelect
      {...props}
      components={Object.assign(
        {DropdownIndicator, GroupHeading},
        props.components
      )}
      styles={styles}
    />
  );
};
