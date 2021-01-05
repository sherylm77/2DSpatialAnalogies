// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/popcode"
	"github.com/emer/etable/etensor"
)

// ExEnv is an example environment, that sets a single input point in a 2D
// input state and two output states as the X and Y coordinates of point.
// It can be used as a starting point for writing your own Env, without
// having much existing code to rewrite.
type ExEnv struct {
	Nm         string `desc:"name of this environment"`
	Dsc        string `desc:"description of this environment"`
	Size       int    `desc:"size of each dimension in 2D input"`
	MinDist    float32
	MaxDist    float32
	MinInp     float32
	MaxInp     float32
	NDistUnits int
	DistPop    popcode.OneD `desc:"population encoding of distance value"`
	Input1Pop  popcode.OneD
	Input2Pop  popcode.OneD
	Input1     etensor.Float32
	Input2     etensor.Float32
	Distance   etensor.Float32
	DistVal    float32
	Inp1Val    float32
	Inp2Val    float32
	HipTable   map[string]etensor.Float32
	Face1      string
	Face2      string
	Run        env.Ctr `view:"inline" desc:"current run of model as provided during Init"`
	Epoch      env.Ctr `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial      env.Ctr `view:"inline" desc:"trial increments over input states -- could add Event as a lower level"`
}

func (ev *ExEnv) Name() string { return ev.Nm }
func (ev *ExEnv) Desc() string { return ev.Dsc }

// Config sets the size, number of trials to run per epoch, and configures the states
func (ev *ExEnv) Config(sz int, ntrls int) {
	ev.Size = sz
	ev.MaxDist = float32(sz)
	ev.MinDist = float32(-1*sz - 2)
	ev.NDistUnits = 10
	ev.DistPop.Defaults()
	ev.DistPop.Min = ev.MinDist
	ev.DistPop.Max = ev.MaxDist + 2

	ev.MaxInp = float32(sz)
	ev.MinInp = -2
	ev.Input1Pop.Defaults()
	ev.Input2Pop.Defaults()
	ev.Input1Pop.Min = ev.MinInp
	ev.Input1Pop.Max = float32(ev.MaxInp) * 1.3
	ev.Input2Pop.Min = ev.MinInp
	ev.Input2Pop.Max = float32(ev.MaxInp) * 1.3
	ev.Input1Pop.Sigma = 0.1
	ev.Input2Pop.Sigma = 0.1
	ev.DistPop.Sigma = 0.1

	currentTime := time.Now()
	rand.Seed(int64(currentTime.Unix()))

	ev.Trial.Max = ntrls

	ev.Distance.SetShape([]int{ev.NDistUnits}, nil, []string{"Distance"})
	ev.Input1.SetShape([]int{sz}, nil, []string{"Input1"})
	ev.Input2.SetShape([]int{sz}, nil, []string{"Input2"})

	ev.HipTable = make(map[string]etensor.Float32)
	var a etensor.Float32
	ev.HipTable["A"] = a
	var b etensor.Float32
	ev.HipTable["B"] = b
	var c etensor.Float32
	ev.HipTable["C"] = c
	var d etensor.Float32
	ev.HipTable["D"] = d

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
		// {"X", []int{ev.Size}, []string{"X"}},
		// {"Y", []int{ev.Size}, []string{"Y"}},
		{"Distance", []int{ev.Size}, []string{"Distance"}},
		{"Input 1", []int{ev.Size}, []string{"Input 1"}},
		{"Input 2", []int{ev.Size}, []string{"Input 2"}},
	}
	return els
}

func (ev *ExEnv) State(element string) etensor.Tensor {
	switch element {
	case "Distance":
		return &ev.Distance
	case "Input 1":
		return &ev.Input1
	case "Input 2":
		return &ev.Input2
	}
	return nil
}

func (ev *ExEnv) Actions() env.Elements {
	return nil
}

// String returns the current state as a string
func (ev *ExEnv) String(isInputTarget bool, whichInput int) string {
	state := ""
	if isInputTarget {
		if whichInput == 1 {
			state = "Input 1 Target"
		} else {
			state = "Input 2 Target"
		}
	} else {
		state = "Distance Target"
	}
	state += " Face " + ev.Face1 + " Face " + ev.Face2
	return fmt.Sprintf(state)
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
	input1 := rand.Intn(int(4)) //rand.Intn(int(ev.MaxInp))
	input2 := rand.Intn(int(4)) //rand.Intn(int(ev.MaxInp))
	distance := input2 - input1
	switch input1 {
	case 0:
		ev.Face1 = "A"
	case 1:
		ev.Face1 = "B"
	case 2:
		ev.Face1 = "C"
	case 3:
		ev.Face1 = "D"
	}
	switch input2 {
	case 0:
		ev.Face2 = "A"
	case 1:
		ev.Face2 = "B"
	case 2:
		ev.Face2 = "C"
	case 3:
		ev.Face2 = "D"
	}

	ev.Input1Pop.Encode(&ev.Input1.Values, float32(input1), int(ev.MaxInp), false)
	ev.Input2Pop.Encode(&ev.Input2.Values, float32(input2), int(ev.MaxInp), false)
	ev.DistPop.Encode(&ev.Distance.Values, float32(distance), ev.NDistUnits, false)
	ev.DistVal = float32(distance)
	ev.Inp1Val = float32(input1)
	ev.Inp2Val = float32(input2)
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
