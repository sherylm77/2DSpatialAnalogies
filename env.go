// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
	"math"
	"math/rand"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/popcode"
	"github.com/emer/etable/etensor"
)

// ExEnv is an example environment, that sets a single input point in a 2D
// input state and two output states as the X and Y coordinates of point.
// It can be used as a starting point for writing your own Env, without
// having much existing code to rewrite.
type ExEnv struct {
	Nm       string `desc:"name of this environment"`
	Dsc      string `desc:"description of this environment"`
	Size     int    `desc:"size of each dimension in 2D input"`
	MaxDist  int
	MaxAngle int
	DistPop  popcode.OneD `desc:"population encoding of distance value"`
	AnglePop popcode.OneD
	Point    image.Point `desc:"X,Y coordinates of point"`
	Point2   image.Point
	Input    etensor.Float32 `desc:"input state, 2D Size x Size"`
	X        etensor.Float32 `desc:"X as a one-hot state 1D Size"`  //Make this distance
	Y        etensor.Float32 `desc:"Y  as a one-hot state 1D Size"` //Make this angle
	Distance etensor.Float32
	Angle    etensor.Float32
	Run      env.Ctr `view:"inline" desc:"current run of model as provided during Init"`
	Epoch    env.Ctr `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial    env.Ctr `view:"inline" desc:"trial increments over input states -- could add Event as a lower level"`
}

func (ev *ExEnv) Name() string { return ev.Nm }
func (ev *ExEnv) Desc() string { return ev.Dsc }

// Config sets the size, number of trials to run per epoch, and configures the states
func (ev *ExEnv) Config(sz int, ntrls int) {
	ev.Size = sz
	ev.MaxDist = int(float64(sz) * math.Sqrt(2))
	ev.MaxAngle = 10 //scaled down from 180
	ev.DistPop.Defaults()
	ev.DistPop.Min = 0
	ev.DistPop.Max = float32(ev.MaxDist) * 1.1
	ev.AnglePop.Defaults()
	ev.AnglePop.Min = -float32(ev.MaxAngle) * 1.1
	ev.AnglePop.Max = float32(ev.MaxAngle) * 1.1
	ev.Trial.Max = ntrls
	ev.Input.SetShape([]int{sz, sz}, nil, []string{"Y", "X"})
	ev.X.SetShape([]int{sz}, nil, []string{"X"})
	ev.Y.SetShape([]int{sz}, nil, []string{"Y"})
	ev.Distance.SetShape([]int{ev.MaxDist}, nil, []string{"Distance"})
	ev.Angle.SetShape([]int{ev.MaxAngle}, nil, []string{"Angle"})
}

func (ev *ExEnv) Validate() error {
	if ev.Size == 0 {
		return fmt.Errorf("ExEnv: %v has size == 0 -- need to Config", ev.Nm)
	}
	return nil
}

func (ev *ExEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Trial}
}

func (ev *ExEnv) States() env.Elements {
	els := env.Elements{
		{"Input", []int{ev.Size, ev.Size}, []string{"Y", "X"}},
		{"X", []int{ev.Size}, []string{"X"}},
		{"Y", []int{ev.Size}, []string{"Y"}},
		{"Distance", []int{ev.Size}, []string{"Distance"}},
		{"Angle", []int{ev.Size}, []string{"Angle"}},
	}
	return els
}

func (ev *ExEnv) State(element string) etensor.Tensor {
	switch element {
	case "Input":
		return &ev.Input
	case "X":
		return &ev.X
	case "Y":
		return &ev.Y
	case "Distance":
		return &ev.Distance
	case "Angle":
		return &ev.Angle
	}
	return nil
}

func (ev *ExEnv) Actions() env.Elements {
	return nil
}

// String returns the current state as a string
func (ev *ExEnv) String() string {
	return fmt.Sprintf("Pt_%d_%d", ev.Point.X, ev.Point.Y)
}

// Init is called to restart environment
func (ev *ExEnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
}

// NewPoint generates a new point and sets state accordingly
func (ev *ExEnv) NewPoint() {
	ev.Point.X = rand.Intn(ev.Size)
	ev.Point.Y = rand.Intn(ev.Size)
	ev.Point2.X = rand.Intn(ev.Size)
	ev.Point2.Y = rand.Intn(ev.Size)
	dist := math.Hypot(float64(ev.Point.X-ev.Point2.X), float64(ev.Point.Y-ev.Point2.Y))
	ang := float64(ev.MaxAngle) * (math.Atan2(float64(ev.Point2.Y-ev.Point.Y), float64(ev.Point2.X-ev.Point.X)) / math.Pi)
	// fmt.Printf("ang: %v\n", ang)
	ev.Input.SetZeros()
	ev.Input.SetFloat([]int{ev.Point.Y, ev.Point.X}, 1)
	ev.Input.SetFloat([]int{ev.Point2.Y, ev.Point2.X}, 1)
	ev.X.SetZeros()
	ev.X.SetFloat([]int{ev.Point.X}, 1)
	ev.X.SetFloat([]int{ev.Point2.X}, 1)
	ev.Y.SetZeros()
	ev.Y.SetFloat([]int{ev.Point.Y}, 1)
	ev.Y.SetFloat([]int{ev.Point2.Y}, 1)
	ev.DistPop.Encode(&ev.Distance.Values, float32(dist), ev.MaxDist)
	ev.AnglePop.Encode(&ev.Angle.Values, float32(ang), ev.MaxAngle)
}

// Step is called to advance the environment state
func (ev *ExEnv) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start
	ev.NewPoint()
	if ev.Trial.Incr() { // true if wraps around Max back to 0
		ev.Epoch.Incr()
	}
	return true
}

func (ev *ExEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *ExEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*ExEnv)(nil)
