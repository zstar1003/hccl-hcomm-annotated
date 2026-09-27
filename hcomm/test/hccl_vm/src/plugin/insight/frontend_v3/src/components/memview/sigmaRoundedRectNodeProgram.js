/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

import { NodeProgram } from 'sigma/rendering'
import { floatColor } from 'sigma/utils'

const TASK_NODE_LABEL_PADDING_X = 6
const TASK_NODE_LABEL_FONT_SIZE = 10
const TASK_NODE_LABEL_MIN_FONT_SIZE = 3
const TASK_NODE_LABEL_MAX_FONT_SIZE = 22
const HOVER_LABEL_PADDING_X = 8
const HOVER_LABEL_HEIGHT = 24
const HOVER_LABEL_RADIUS = 6

function roundedRectPath(context, x, y, width, height, radius) {
  const safeRadius = Math.max(0, Math.min(radius, width / 2, height / 2))
  context.beginPath()
  context.moveTo(x + safeRadius, y)
  context.arcTo(x + width, y, x + width, y + height, safeRadius)
  context.arcTo(x + width, y + height, x, y + height, safeRadius)
  context.arcTo(x, y + height, x, y, safeRadius)
  context.arcTo(x, y, x + width, y, safeRadius)
  context.closePath()
}

function truncateLabelToWidth(context, label, maxWidth) {
  if (!label) return ''
  if (maxWidth <= 8) return label
  if (context.measureText(label).width <= maxWidth) return label

  const ellipsis = '...'
  let end = label.length
  while (end > 1) {
    const next = `${label.slice(0, end)}${ellipsis}`
    if (context.measureText(next).width <= maxWidth) return next
    end -= 1
  }
  return ellipsis
}

const VERTEX_SHADER_SOURCE = /* glsl */ `
attribute vec4 a_id;
attribute vec4 a_fillColor;
attribute vec4 a_borderColor;
attribute vec2 a_position;
attribute float a_size;
attribute vec2 a_dimensions;
attribute float a_border;
attribute float a_corner;

uniform float u_sizeRatio;
uniform float u_pixelRatio;
uniform mat3 u_matrix;

varying vec4 v_fillColor;
varying vec4 v_borderColor;
varying vec2 v_dimensions;
varying float v_border;
varying float v_corner;
varying float v_feather;

const float bias = 255.0 / 254.0;

void main() {
  gl_Position = vec4(
    (u_matrix * vec3(a_position, 1.0)).xy,
    0.0,
    1.0
  );

  gl_PointSize = a_size / u_sizeRatio * u_pixelRatio * 2.0;

  v_dimensions = a_dimensions;
  v_border = a_border;
  v_corner = a_corner;
  v_feather = max(1.0, u_pixelRatio) * 1.5 / max(gl_PointSize, 1.0);

  #ifdef PICKING_MODE
  v_fillColor = a_id;
  v_borderColor = a_id;
  #else
  v_fillColor = a_fillColor;
  v_borderColor = a_borderColor;
  #endif

  v_fillColor.a *= bias;
  v_borderColor.a *= bias;
}
`

const FRAGMENT_SHADER_SOURCE = /* glsl */ `
precision mediump float;

varying vec4 v_fillColor;
varying vec4 v_borderColor;
varying vec2 v_dimensions;
varying float v_border;
varying float v_corner;
varying float v_feather;

const vec4 transparent = vec4(0.0, 0.0, 0.0, 0.0);

float roundedRectSdf(vec2 point, vec2 halfSize, float radius) {
  vec2 q = abs(point) - (halfSize - vec2(radius));
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

void main(void) {
  vec2 point = gl_PointCoord - vec2(0.5, 0.5);
  vec2 halfSize = max(v_dimensions * 0.5, vec2(0.0001, 0.0001));
  float radius = min(v_corner, min(halfSize.x, halfSize.y));
  float border = max(0.0, min(v_border, min(halfSize.x, halfSize.y)));
  float dist = roundedRectSdf(point, halfSize, radius);

  #ifdef PICKING_MODE
  if (dist <= 0.0)
    gl_FragColor = v_fillColor;
  else
    gl_FragColor = transparent;
  #else
  float outerAlpha = 1.0 - smoothstep(0.0, v_feather, dist);
  float innerAlpha = 1.0 - smoothstep(0.0, v_feather, dist + border);
  vec4 color = mix(v_borderColor, v_fillColor, innerAlpha);
  float alpha = color.a * outerAlpha;
  gl_FragColor = vec4(color.rgb * alpha, alpha);
  #endif
}
`

const UNIFORMS = ['u_sizeRatio', 'u_pixelRatio', 'u_matrix']

export default class SigmaRoundedRectNodeProgram extends NodeProgram {
  drawHover = (context, data, settings) => {
    if (!data?.isTaskNode || !data.label) return

    const widthFactor = Number(data.rectWidthFactor ?? 1)
    const heightFactor = Number(data.rectHeightFactor ?? 1)
    const rectWidth = Math.max(1, Number(data.size ?? 0) * 2 * widthFactor)
    const rectHeight = Math.max(1, Number(data.size ?? 0) * 2 * heightFactor)
    const label = String(data.label)
    const fontSize = Math.max(11, Math.min(14, rectHeight * 0.42))
    const inlineFontSize = Math.max(
      TASK_NODE_LABEL_MIN_FONT_SIZE,
      Math.min(TASK_NODE_LABEL_MAX_FONT_SIZE, rectHeight * (TASK_NODE_LABEL_FONT_SIZE / 24)),
    )

    context.save()
    context.font = `600 ${fontSize}px ${settings.labelFont}`
    const textWidth = context.measureText(label).width
    const boxWidth = Math.max(rectWidth, textWidth + HOVER_LABEL_PADDING_X * 2)
    const boxHeight = HOVER_LABEL_HEIGHT
    const boxX = data.x - boxWidth / 2
    const boxY = data.y - rectHeight / 2 - boxHeight - 8

    roundedRectPath(context, boxX, boxY, boxWidth, boxHeight, HOVER_LABEL_RADIUS)
    context.fillStyle = 'rgba(12, 22, 38, 0.96)'
    context.fill()
    context.strokeStyle = data.borderColor || '#7ec0ff'
    context.lineWidth = 1.2
    context.stroke()

    context.fillStyle = '#f3f7ff'
    context.textAlign = 'center'
    context.textBaseline = 'middle'
    context.fillText(label, data.x, boxY + boxHeight / 2 + 0.5)

    context.font = `600 ${inlineFontSize}px ${settings.labelFont}`
    context.textAlign = 'left'
    const inlineLabelWidth = Math.max(1, rectWidth - TASK_NODE_LABEL_PADDING_X * 2)
    const inlineLabel = truncateLabelToWidth(context, label, inlineLabelWidth)
    context.fillText(
      inlineLabel,
      data.x - rectWidth / 2 + TASK_NODE_LABEL_PADDING_X,
      data.y,
      inlineLabelWidth,
    )
    context.restore()
  }

  drawLabel = (context, data, settings) => {
    if (!data?.isTaskNode || !data.label) return

    const widthFactor = Number(data.rectWidthFactor ?? 1)
    const heightFactor = Number(data.rectHeightFactor ?? 1)
    const rectWidth = Math.max(1, Number(data.size ?? 0) * 2 * widthFactor)
    const rectHeight = Math.max(1, Number(data.size ?? 0) * 2 * heightFactor)
    const scaledFontSize = Math.max(
      TASK_NODE_LABEL_MIN_FONT_SIZE,
      Math.min(TASK_NODE_LABEL_MAX_FONT_SIZE, rectHeight * (TASK_NODE_LABEL_FONT_SIZE / 24)),
    )

    context.save()
    context.font = `600 ${scaledFontSize}px ${settings.labelFont}`
    context.fillStyle = '#f3f7ff'
    context.textAlign = 'left'
    context.textBaseline = 'middle'

    const maxLabelWidth = Math.max(1, rectWidth - TASK_NODE_LABEL_PADDING_X * 2)
    const label = truncateLabelToWidth(context, data.label, maxLabelWidth)
    context.fillText(
      label,
      data.x - rectWidth / 2 + TASK_NODE_LABEL_PADDING_X,
      data.y,
      maxLabelWidth,
    )
    context.restore()
  }

  getDefinition() {
    return {
      VERTICES: 1,
      VERTEX_SHADER_SOURCE,
      FRAGMENT_SHADER_SOURCE,
      METHOD: WebGLRenderingContext.POINTS,
      UNIFORMS,
      ATTRIBUTES: [
        { name: 'a_position', size: 2, type: WebGLRenderingContext.FLOAT },
        { name: 'a_size', size: 1, type: WebGLRenderingContext.FLOAT },
        { name: 'a_dimensions', size: 2, type: WebGLRenderingContext.FLOAT },
        { name: 'a_border', size: 1, type: WebGLRenderingContext.FLOAT },
        { name: 'a_corner', size: 1, type: WebGLRenderingContext.FLOAT },
        {
          name: 'a_fillColor',
          size: 4,
          type: WebGLRenderingContext.UNSIGNED_BYTE,
          normalized: true,
        },
        {
          name: 'a_borderColor',
          size: 4,
          type: WebGLRenderingContext.UNSIGNED_BYTE,
          normalized: true,
        },
        {
          name: 'a_id',
          size: 4,
          type: WebGLRenderingContext.UNSIGNED_BYTE,
          normalized: true,
        },
      ],
    }
  }

  processVisibleItem(nodeIndex, startIndex, data) {
    const maxDimension = Math.max(Number(data.rectWidth) || 1, Number(data.rectHeight) || 1, 1)
    const widthFactor = Math.max(0.02, (Number(data.rectWidth) || maxDimension) / maxDimension)
    const heightFactor = Math.max(0.02, (Number(data.rectHeight) || maxDimension) / maxDimension)
    const borderFactor = Math.max(0.002, (Number(data.rectBorderWidth) || 1) / maxDimension)
    const cornerFactor = Math.max(0.002, (Number(data.rectCornerRadius) || 0) / maxDimension)

    const array = this.array
    array[startIndex++] = data.x
    array[startIndex++] = data.y
    array[startIndex++] = data.size
    array[startIndex++] = widthFactor
    array[startIndex++] = heightFactor
    array[startIndex++] = borderFactor
    array[startIndex++] = cornerFactor
    array[startIndex++] = floatColor(data.fillColor || data.color)
    array[startIndex++] = floatColor(data.borderColor || data.color)
    array[startIndex++] = nodeIndex
  }

  setUniforms(params, { gl, uniformLocations }) {
    gl.uniform1f(uniformLocations.u_sizeRatio, params.sizeRatio)
    gl.uniform1f(uniformLocations.u_pixelRatio, params.pixelRatio)
    gl.uniformMatrix3fv(uniformLocations.u_matrix, false, params.matrix)
  }
}
