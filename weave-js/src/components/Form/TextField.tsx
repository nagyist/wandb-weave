/**
 * A form text input.
 *
 * Design:
 * https://www.figma.com/file/01KWBdMZg5QM9SRS1pQq0z/Design-System----Robot-Styles?type=design&node-id=92%3A1442&t=141heo3Rv8zF83xH-1
 */

import {Icon, IconName} from '@wandb/weave/components/Icon';
import {Tailwind} from '@wandb/weave/components/Tailwind';
import classNames from 'classnames';
import React from 'react';

export const TextFieldSizes = {
  Small: 'small',
  Medium: 'medium',
  Large: 'large',
} as const;
export type TextFieldSize =
  (typeof TextFieldSizes)[keyof typeof TextFieldSizes];

type TextFieldProps = {
  /** id to assign to the input element */
  id?: string;
  size?: TextFieldSize;
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  onKeyDown?: (key: string, e: React.KeyboardEvent<HTMLInputElement>) => void;
  onBlur?: (value: string) => void;
  onFocus?: () => void;
  autoFocus?: boolean;
  disabled?: boolean;
  icon?: IconName;
  ariaLabel?: string;
  errorState?: boolean;
  prefix?: string;
  extraActions?: React.ReactNode;
  maxLength?: number;
  type?: string;
  autoComplete?: string;
  dataTest?: string;
  step?: number;
  variant?: 'default' | 'ghost';
  isContainerNightAware?: boolean;
};

export const TextField = ({
  id,
  size,
  variant = 'default',
  placeholder,
  value,
  onChange,
  onKeyDown,
  onBlur,
  onFocus,
  autoFocus,
  disabled,
  icon,
  ariaLabel,
  errorState,
  prefix,
  extraActions,
  maxLength,
  type,
  autoComplete,
  dataTest,
  step,
  isContainerNightAware,
}: TextFieldProps) => {
  const textFieldSize = size ?? 'medium';
  const leftPaddingForIcon =
    textFieldSize === 'small'
      ? 'pl-32'
      : textFieldSize === 'medium'
      ? 'pl-34'
      : 'pl-36';

  const handleChange = onChange
    ? (e: React.ChangeEvent<HTMLInputElement>) => {
        onChange(e.target.value);
      }
    : undefined;
  const handleKeyDown = onKeyDown
    ? (e: React.KeyboardEvent<HTMLInputElement>) => {
        onKeyDown(e.key, e);
      }
    : undefined;
  const handleBlur = onBlur
    ? (e: React.ChangeEvent<HTMLInputElement>) => {
        onBlur?.(e.target.value);
      }
    : undefined;

  return (
    <Tailwind style={{width: '100%'}}>
      <div
        className={classNames(
          'relative rounded-[5px]',
          textFieldSize === 'small'
            ? 'h-30'
            : textFieldSize === 'medium'
            ? 'h-32'
            : 'h-40',
          'bg-white dark:bg-moon-900',
          'text-moon-800 dark:text-moon-200',
          variant === 'default' &&
            'outline outline-1 outline-moon-250 dark:outline-moon-700',
          variant === 'ghost' &&
            'outline outline-1 outline-transparent dark:outline-transparent',
          {
            // must not add "night-aware" class if already in a night-aware
            // container, otherwise they'll cancel each other out
            'night-aware': !isContainerNightAware,
            'hover:outline-2 [&:hover:not(:focus-within)]:outline-[#83E4EB] dark:[&:hover:not(:focus-within)]:outline-teal-650':
              !errorState && variant === 'default',
            'focus-within:outline-2 focus-within:outline-teal-400 dark:focus-within:outline-teal-600':
              !errorState && variant === 'default',
            'outline-2 outline-red-450 dark:outline-red-550':
              errorState && variant === 'default',
            'pointer-events-none opacity-50': disabled,
          }
        )}>
        <div className="absolute bottom-0 top-0 flex w-full items-center rounded-[5px]">
          {prefix && (
            <div
              className={classNames(
                'text-gray-800 pr-1',
                icon ? leftPaddingForIcon : 'pl-8'
              )}>
              {prefix}
            </div>
          )}
          <input
            id={id}
            className={classNames(
              'h-full w-full flex-1 rounded-[5px] bg-inherit pr-8',
              'appearance-none border-none',
              'focus:outline-none',
              'placeholder-moon-500',
              'dark:selection:bg-moon-650 dark:selection:text-moon-200',
              {
                [leftPaddingForIcon]: icon && !prefix,
                'pl-8': !icon && !prefix,
              }
            )}
            placeholder={placeholder}
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            onBlur={handleBlur}
            onFocus={onFocus}
            autoFocus={autoFocus}
            disabled={disabled}
            readOnly={!onChange} // It would be readonly regardless but this prevents a console warning
            aria-label={ariaLabel}
            maxLength={maxLength}
            type={type}
            autoComplete={autoComplete}
            data-test={dataTest}
            step={step}
          />
          {extraActions}
        </div>

        {icon && (
          <Icon
            name={icon}
            className={classNames(
              'absolute left-8',
              textFieldSize === 'small'
                ? 'top-6 h-16 w-16'
                : textFieldSize === 'medium'
                ? 'top-8 h-18 w-18'
                : 'top-10 h-20 w-20',
              value ? 'text-moon-800 dark:text-moon-200' : 'text-moon-500'
            )}
          />
        )}
      </div>
    </Tailwind>
  );
};
