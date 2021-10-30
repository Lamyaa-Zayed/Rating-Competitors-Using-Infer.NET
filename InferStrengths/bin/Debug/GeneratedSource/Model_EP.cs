// <auto-generated />
#pragma warning disable 1570, 1591

using System;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;

namespace Models
{
	/// <summary>
	/// Generated algorithm for performing inference.
	/// </summary>
	/// <remarks>
	/// If you wish to use this class directly, you must perform the following steps:
	/// 1) Create an instance of the class.
	/// 2) Set the value of any externally-set fields e.g. data, priors.
	/// 3) Call the Execute(numberOfIterations) method.
	/// 4) Use the XXXMarginal() methods to retrieve posterior marginals for different variables.
	/// 
	/// Generated by Infer.NET 0.3.2102.1701 at 7:20 PM on Saturday, October 30, 2021.
	/// </remarks>
	public partial class Model_EP : IGeneratedAlgorithm
	{
		#region Fields
		/// <summary>True if Changed_losers has executed. Set this to false to force re-execution of Changed_losers</summary>
		public bool Changed_losers_isDone;
		/// <summary>True if Changed_losers_numberOfIterations_winners has executed. Set this to false to force re-execution of Changed_losers_numberOfIterations_winners</summary>
		public bool Changed_losers_numberOfIterations_winners_isDone;
		/// <summary>True if Changed_numberOfIterationsDecreased_Init_losers_winners has executed. Set this to false to force re-execution of Changed_numberOfIterationsDecreased_Init_losers_winners</summary>
		public bool Changed_numberOfIterationsDecreased_Init_losers_winners_isDone;
		/// <summary>True if Changed_numberOfIterationsDecreased_Init_losers_winners has performed initialisation. Set this to false to force re-execution of Changed_numberOfIterationsDecreased_Init_losers_winners</summary>
		public bool Changed_numberOfIterationsDecreased_Init_losers_winners_isInitialised;
		/// <summary>True if Changed_winners has executed. Set this to false to force re-execution of Changed_winners</summary>
		public bool Changed_winners_isDone;
		/// <summary>True if Constant has executed. Set this to false to force re-execution of Constant</summary>
		public bool Constant_isDone;
		/// <summary>Field backing the losers property</summary>
		private int[] Losers;
		/// <summary>Message to marginal of 'losers'</summary>
		public PointMass<int[]> losers_marginal_F;
		/// <summary>Field backing the NumberOfIterationsDone property</summary>
		private int numberOfIterationsDone;
		public DistributionStructArray<Gaussian,double> strengths_itemlosers_gameIDsRange__B;
		public DistributionStructArray<Gaussian,double> strengths_itemwinners_gameIDsRange__B;
		/// <summary>Message to marginal of 'strengths'</summary>
		public DistributionStructArray<Gaussian,double> strengths_marginal_F;
		/// <summary>Messages from uses of 'strengths_use'</summary>
		public DistributionStructArray<Gaussian,double>[] strengths_uses_B;
		/// <summary>Field backing the winners property</summary>
		private int[] Winners;
		/// <summary>Message to marginal of 'winners'</summary>
		public PointMass<int[]> winners_marginal_F;
		#endregion

		#region Properties
		/// <summary>The externally-specified value of 'losers'</summary>
		public int[] losers
		{
			get {
				return this.Losers;
			}
			set {
				if ((value!=null)&&(value.Length!=9)) {
					throw new ArgumentException(((("Provided array of length "+value.Length)+" when length ")+9)+" was expected for variable \'losers\'");
				}
				this.Losers = value;
				this.numberOfIterationsDone = 0;
				this.Changed_losers_isDone = false;
				this.Changed_numberOfIterationsDecreased_Init_losers_winners_isInitialised = false;
				this.Changed_losers_numberOfIterations_winners_isDone = false;
			}
		}

		/// <summary>The number of iterations done from the initial state</summary>
		public int NumberOfIterationsDone
		{
			get {
				return this.numberOfIterationsDone;
			}
		}

		/// <summary>The externally-specified value of 'winners'</summary>
		public int[] winners
		{
			get {
				return this.Winners;
			}
			set {
				if ((value!=null)&&(value.Length!=9)) {
					throw new ArgumentException(((("Provided array of length "+value.Length)+" when length ")+9)+" was expected for variable \'winners\'");
				}
				this.Winners = value;
				this.numberOfIterationsDone = 0;
				this.Changed_winners_isDone = false;
				this.Changed_numberOfIterationsDecreased_Init_losers_winners_isInitialised = false;
				this.Changed_losers_numberOfIterations_winners_isDone = false;
			}
		}

		#endregion

		#region Methods
		/// <summary>Computations that depend on the observed value of losers</summary>
		private void Changed_losers()
		{
			if (this.Changed_losers_isDone) {
				return ;
			}
			// Create array for 'losers_marginal' Forwards messages.
			this.losers_marginal_F = new PointMass<int[]>(this.Losers);
			// Message to 'losers_marginal' from DerivedVariable factor
			this.losers_marginal_F = DerivedVariableOp.MarginalAverageConditional<PointMass<int[]>,int[]>(this.Losers, this.losers_marginal_F);
			this.Changed_losers_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of losers and numberOfIterations and winners</summary>
		/// <param name="numberOfIterations">The number of times to iterate each loop</param>
		private void Changed_losers_numberOfIterations_winners(int numberOfIterations)
		{
			if (this.Changed_losers_numberOfIterations_winners_isDone) {
				return ;
			}
			DistributionStructArray<Gaussian,double> strengths_F;
			Gaussian strengths_F_reduced;
			// Create array for 'strengths' Forwards messages.
			strengths_F = new DistributionStructArray<Gaussian,double>(6);
			// Message to 'strengths' from GaussianFromMeanAndVariance factor
			strengths_F_reduced = GaussianFromMeanAndVarianceOp.SampleAverageConditional(2000.0, 40000.0);
			for(int teamsIDRange = 0; teamsIDRange<6; teamsIDRange++) {
				strengths_F[teamsIDRange] = strengths_F_reduced;
				strengths_F[teamsIDRange] = strengths_F_reduced;
			}
			// Create array for 'strengths_marginal' Forwards messages.
			this.strengths_marginal_F = new DistributionStructArray<Gaussian,double>(6);
			DistributionStructArray<Gaussian,double> strengths_use_B;
			// Create array for 'strengths_use' Backwards messages.
			strengths_use_B = new DistributionStructArray<Gaussian,double>(6);
			for(int teamsIDRange = 0; teamsIDRange<6; teamsIDRange++) {
				strengths_use_B[teamsIDRange] = Gaussian.Uniform();
			}
			DistributionStructArray<Gaussian,double>[] strengths_uses_F;
			// Create array for 'strengths_uses' Forwards messages.
			strengths_uses_F = new DistributionStructArray<Gaussian,double>[2];
			// Create array for 'strengths_uses' Forwards messages.
			strengths_uses_F[1] = new DistributionStructArray<Gaussian,double>(6);
			for(int teamsIDRange = 0; teamsIDRange<6; teamsIDRange++) {
				strengths_uses_F[1][teamsIDRange] = Gaussian.Uniform();
			}
			DistributionStructArray<Gaussian,double> strengths_uses_F_1__marginal;
			// Message to 'strengths_itemlosers_gameIDsRange_' from GetItems factor
			strengths_uses_F_1__marginal = GetItemsOp<double>.MarginalInit<DistributionStructArray<Gaussian,double>>(strengths_uses_F[1]);
			DistributionStructArray<Gaussian,double> strengths_itemlosers_gameIDsRange__F;
			// Create array for 'strengths_itemlosers_gameIDsRange_' Forwards messages.
			strengths_itemlosers_gameIDsRange__F = new DistributionStructArray<Gaussian,double>(9);
			for(int gameIDsRange = 0; gameIDsRange<9; gameIDsRange++) {
				strengths_itemlosers_gameIDsRange__F[gameIDsRange] = Gaussian.Uniform();
			}
			// Create array for replicates of 'losePerf_F'
			Gaussian[] losePerf_F = new Gaussian[9];
			for(int gameIDsRange = 0; gameIDsRange<9; gameIDsRange++) {
				losePerf_F[gameIDsRange] = Gaussian.Uniform();
			}
			// Create array for 'strengths_uses' Forwards messages.
			strengths_uses_F[0] = new DistributionStructArray<Gaussian,double>(6);
			for(int teamsIDRange = 0; teamsIDRange<6; teamsIDRange++) {
				strengths_uses_F[0][teamsIDRange] = Gaussian.Uniform();
			}
			DistributionStructArray<Gaussian,double> strengths_uses_F_0__marginal;
			// Message to 'strengths_itemwinners_gameIDsRange_' from GetItems factor
			strengths_uses_F_0__marginal = GetItemsOp<double>.MarginalInit<DistributionStructArray<Gaussian,double>>(strengths_uses_F[0]);
			DistributionStructArray<Gaussian,double> strengths_itemwinners_gameIDsRange__F;
			// Create array for 'strengths_itemwinners_gameIDsRange_' Forwards messages.
			strengths_itemwinners_gameIDsRange__F = new DistributionStructArray<Gaussian,double>(9);
			for(int gameIDsRange = 0; gameIDsRange<9; gameIDsRange++) {
				strengths_itemwinners_gameIDsRange__F[gameIDsRange] = Gaussian.Uniform();
			}
			// Create array for replicates of 'winPerf_F'
			Gaussian[] winPerf_F = new Gaussian[9];
			for(int gameIDsRange = 0; gameIDsRange<9; gameIDsRange++) {
				winPerf_F[gameIDsRange] = Gaussian.Uniform();
			}
			// Create array for replicates of 'vdouble12_F'
			Gaussian[] vdouble12_F = new Gaussian[9];
			for(int gameIDsRange = 0; gameIDsRange<9; gameIDsRange++) {
				vdouble12_F[gameIDsRange] = Gaussian.Uniform();
			}
			// Create array for replicates of 'vdouble12_B'
			Gaussian[] vdouble12_B = new Gaussian[9];
			for(int gameIDsRange = 0; gameIDsRange<9; gameIDsRange++) {
				vdouble12_B[gameIDsRange] = Gaussian.Uniform();
			}
			// Create array for replicates of 'winPerf_use_B'
			Gaussian[] winPerf_use_B = new Gaussian[9];
			for(int gameIDsRange = 0; gameIDsRange<9; gameIDsRange++) {
				winPerf_use_B[gameIDsRange] = Gaussian.Uniform();
			}
			// Create array for replicates of 'losePerf_use_B'
			Gaussian[] losePerf_use_B = new Gaussian[9];
			for(int gameIDsRange = 0; gameIDsRange<9; gameIDsRange++) {
				losePerf_use_B[gameIDsRange] = Gaussian.Uniform();
			}
			for(int iteration = this.numberOfIterationsDone; iteration<numberOfIterations; iteration++) {
				// Message to 'strengths_uses' from Replicate factor
				strengths_uses_F[1] = ReplicateOp_NoDivide.UsesAverageConditional<DistributionStructArray<Gaussian,double>>(this.strengths_uses_B, strengths_F, 1, strengths_uses_F[1]);
				// Message to 'strengths_itemlosers_gameIDsRange_' from GetItems factor
				strengths_uses_F_1__marginal = GetItemsOp<double>.Marginal<DistributionStructArray<Gaussian,double>,Gaussian>(strengths_uses_F[1], this.strengths_uses_B[1], strengths_uses_F_1__marginal);
				// Message to 'strengths_uses' from Replicate factor
				strengths_uses_F[0] = ReplicateOp_NoDivide.UsesAverageConditional<DistributionStructArray<Gaussian,double>>(this.strengths_uses_B, strengths_F, 0, strengths_uses_F[0]);
				// Message to 'strengths_itemwinners_gameIDsRange_' from GetItems factor
				strengths_uses_F_0__marginal = GetItemsOp<double>.Marginal<DistributionStructArray<Gaussian,double>,Gaussian>(strengths_uses_F[0], this.strengths_uses_B[0], strengths_uses_F_0__marginal);
				for(int gameIDsRange = 0; gameIDsRange<9; gameIDsRange++) {
					// Message to 'strengths_itemlosers_gameIDsRange_' from GetItems factor
					strengths_itemlosers_gameIDsRange__F[gameIDsRange] = GetItemsOp<double>.ItemsAverageConditional<DistributionStructArray<Gaussian,double>,Gaussian>(this.strengths_itemlosers_gameIDsRange__B[gameIDsRange], strengths_uses_F[1], strengths_uses_F_1__marginal, this.Losers, gameIDsRange, strengths_itemlosers_gameIDsRange__F[gameIDsRange]);
					// Message to 'losePerf' from GaussianFromMeanAndVariance factor
					losePerf_F[gameIDsRange] = GaussianFromMeanAndVarianceOp.SampleAverageConditional(strengths_itemlosers_gameIDsRange__F[gameIDsRange], 400.0);
					// Message to 'strengths_itemwinners_gameIDsRange_' from GetItems factor
					strengths_itemwinners_gameIDsRange__F[gameIDsRange] = GetItemsOp<double>.ItemsAverageConditional<DistributionStructArray<Gaussian,double>,Gaussian>(this.strengths_itemwinners_gameIDsRange__B[gameIDsRange], strengths_uses_F[0], strengths_uses_F_0__marginal, this.Winners, gameIDsRange, strengths_itemwinners_gameIDsRange__F[gameIDsRange]);
					// Message to 'winPerf' from GaussianFromMeanAndVariance factor
					winPerf_F[gameIDsRange] = GaussianFromMeanAndVarianceOp.SampleAverageConditional(strengths_itemwinners_gameIDsRange__F[gameIDsRange], 400.0);
					// Message to 'vdouble12' from Difference factor
					vdouble12_F[gameIDsRange] = DoublePlusOp.AAverageConditional(winPerf_F[gameIDsRange], losePerf_F[gameIDsRange]);
					// Message to 'vdouble12' from IsPositive factor
					vdouble12_B[gameIDsRange] = IsPositiveOp_Proper.XAverageConditional(Bernoulli.PointMass(true), vdouble12_F[gameIDsRange]);
					// Message to 'winPerf_use' from Difference factor
					winPerf_use_B[gameIDsRange] = DoublePlusOp.SumAverageConditional(vdouble12_B[gameIDsRange], losePerf_F[gameIDsRange]);
					// Message to 'strengths_itemwinners_gameIDsRange_' from GaussianFromMeanAndVariance factor
					this.strengths_itemwinners_gameIDsRange__B[gameIDsRange] = GaussianFromMeanAndVarianceOp.MeanAverageConditional(winPerf_use_B[gameIDsRange], 400.0);
					// Message to 'losePerf_use' from Difference factor
					losePerf_use_B[gameIDsRange] = DoublePlusOp.BAverageConditional(winPerf_F[gameIDsRange], vdouble12_B[gameIDsRange]);
					// Message to 'strengths_itemlosers_gameIDsRange_' from GaussianFromMeanAndVariance factor
					this.strengths_itemlosers_gameIDsRange__B[gameIDsRange] = GaussianFromMeanAndVarianceOp.MeanAverageConditional(losePerf_use_B[gameIDsRange], 400.0);
				}
				// Message to 'strengths_uses' from GetItems factor
				this.strengths_uses_B[0] = GetItemsOp<double>.ArrayAverageConditional<Gaussian,DistributionStructArray<Gaussian,double>>(this.strengths_itemwinners_gameIDsRange__B, this.Winners, this.strengths_uses_B[0]);
				// Message to 'strengths_uses' from GetItems factor
				this.strengths_uses_B[1] = GetItemsOp<double>.ArrayAverageConditional<Gaussian,DistributionStructArray<Gaussian,double>>(this.strengths_itemlosers_gameIDsRange__B, this.Losers, this.strengths_uses_B[1]);
				this.OnProgressChanged(new ProgressChangedEventArgs(iteration));
			}
			// Message to 'strengths_use' from Replicate factor
			strengths_use_B = ReplicateOp_NoDivide.DefAverageConditional<DistributionStructArray<Gaussian,double>>(this.strengths_uses_B, strengths_use_B);
			for(int teamsIDRange = 0; teamsIDRange<6; teamsIDRange++) {
				this.strengths_marginal_F[teamsIDRange] = Gaussian.Uniform();
				// Message to 'strengths_marginal' from Variable factor
				this.strengths_marginal_F[teamsIDRange] = VariableOp.MarginalAverageConditional<Gaussian>(strengths_use_B[teamsIDRange], strengths_F_reduced, this.strengths_marginal_F[teamsIDRange]);
			}
			this.Changed_losers_numberOfIterations_winners_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of numberOfIterationsDecreased and must reset on changes to losers and winners</summary>
		/// <param name="initialise">If true, reset messages that initialise loops</param>
		private void Changed_numberOfIterationsDecreased_Init_losers_winners(bool initialise)
		{
			if (this.Changed_numberOfIterationsDecreased_Init_losers_winners_isDone&&((!initialise)||this.Changed_numberOfIterationsDecreased_Init_losers_winners_isInitialised)) {
				return ;
			}
			for(int teamsIDRange = 0; teamsIDRange<6; teamsIDRange++) {
				this.strengths_uses_B[0][teamsIDRange] = Gaussian.Uniform();
				this.strengths_uses_B[1][teamsIDRange] = Gaussian.Uniform();
			}
			for(int gameIDsRange = 0; gameIDsRange<9; gameIDsRange++) {
				this.strengths_itemlosers_gameIDsRange__B[gameIDsRange] = Gaussian.Uniform();
				this.strengths_itemwinners_gameIDsRange__B[gameIDsRange] = Gaussian.Uniform();
			}
			this.Changed_numberOfIterationsDecreased_Init_losers_winners_isDone = true;
			this.Changed_numberOfIterationsDecreased_Init_losers_winners_isInitialised = true;
		}

		/// <summary>Computations that depend on the observed value of winners</summary>
		private void Changed_winners()
		{
			if (this.Changed_winners_isDone) {
				return ;
			}
			// Create array for 'winners_marginal' Forwards messages.
			this.winners_marginal_F = new PointMass<int[]>(this.Winners);
			// Message to 'winners_marginal' from DerivedVariable factor
			this.winners_marginal_F = DerivedVariableOp.MarginalAverageConditional<PointMass<int[]>,int[]>(this.Winners, this.winners_marginal_F);
			this.Changed_winners_isDone = true;
		}

		/// <summary>Computations that do not depend on observed values</summary>
		private void Constant()
		{
			if (this.Constant_isDone) {
				return ;
			}
			// Create array for 'strengths_uses' Backwards messages.
			this.strengths_uses_B = new DistributionStructArray<Gaussian,double>[2];
			// Create array for 'strengths_uses' Backwards messages.
			this.strengths_uses_B[0] = new DistributionStructArray<Gaussian,double>(6);
			// Create array for 'strengths_uses' Backwards messages.
			this.strengths_uses_B[1] = new DistributionStructArray<Gaussian,double>(6);
			// Create array for 'strengths_itemlosers_gameIDsRange_' Backwards messages.
			this.strengths_itemlosers_gameIDsRange__B = new DistributionStructArray<Gaussian,double>(9);
			// Create array for 'strengths_itemwinners_gameIDsRange_' Backwards messages.
			this.strengths_itemwinners_gameIDsRange__B = new DistributionStructArray<Gaussian,double>(9);
			bool vbool0_reduced = default(bool);
			vbool0_reduced = true;
			Constrain.Equal<bool>(true, vbool0_reduced);
			this.Constant_isDone = true;
		}

		/// <summary>Update all marginals, by iterating message passing the given number of times</summary>
		/// <param name="numberOfIterations">The number of times to iterate each loop</param>
		/// <param name="initialise">If true, messages that initialise loops are reset when observed values change</param>
		private void Execute(int numberOfIterations, bool initialise)
		{
			if (numberOfIterations!=this.numberOfIterationsDone) {
				if (numberOfIterations<this.numberOfIterationsDone) {
					this.numberOfIterationsDone = 0;
					this.Changed_numberOfIterationsDecreased_Init_losers_winners_isDone = false;
				}
				this.Changed_losers_numberOfIterations_winners_isDone = false;
			}
			this.Changed_losers();
			this.Changed_winners();
			this.Constant();
			this.Changed_numberOfIterationsDecreased_Init_losers_winners(initialise);
			this.Changed_losers_numberOfIterations_winners(numberOfIterations);
			this.numberOfIterationsDone = numberOfIterations;
		}

		/// <summary>Update all marginals, by iterating message-passing the given number of times</summary>
		/// <param name="numberOfIterations">The total number of iterations that should be executed for the current set of observed values.  If this is more than the number already done, only the extra iterations are done.  If this is less than the number already done, message-passing is restarted from the beginning.  Changing the observed values resets the iteration count to 0.</param>
		public void Execute(int numberOfIterations)
		{
			this.Execute(numberOfIterations, true);
		}

		/// <summary>Get the observed value of the specified variable.</summary>
		/// <param name="variableName">Variable name</param>
		public object GetObservedValue(string variableName)
		{
			if (variableName=="winners") {
				return this.winners;
			}
			if (variableName=="losers") {
				return this.losers;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>
		/// Returns the marginal distribution for 'losers' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public PointMass<int[]> LosersMarginal()
		{
			return this.losers_marginal_F;
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName)
		{
			if (variableName=="losers") {
				return this.LosersMarginal();
			}
			if (variableName=="winners") {
				return this.WinnersMarginal();
			}
			if (variableName=="strengths") {
				return this.StrengthsMarginal();
			}
			throw new ArgumentException("This class was not built to infer "+variableName);
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable, converted to type T</summary>
		/// <typeparam name="T">The distribution type.</typeparam>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public T Marginal<T>(string variableName)
		{
			return Distribution.ChangeType<T>(this.Marginal(variableName));
		}

		/// <summary>Get the query-specific marginal distribution of a variable.</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <param name="query">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName, string query)
		{
			if (query=="Marginal") {
				return this.Marginal(variableName);
			}
			throw new ArgumentException(((("This class was not built to infer \'"+variableName)+"\' with query \'")+query)+"\'");
		}

		/// <summary>Get the query-specific marginal distribution of a variable, converted to type T</summary>
		/// <typeparam name="T">The distribution type.</typeparam>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <param name="query">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public T Marginal<T>(string variableName, string query)
		{
			return Distribution.ChangeType<T>(this.Marginal(variableName, query));
		}

		private void OnProgressChanged(ProgressChangedEventArgs e)
		{
			// Make a temporary copy of the event to avoid a race condition
			// if the last subscriber unsubscribes immediately after the null check and before the event is raised.
			EventHandler<ProgressChangedEventArgs> handler = this.ProgressChanged;
			if (handler!=null) {
				handler(this, e);
			}
		}

		/// <summary>Reset all messages to their initial values.  Sets NumberOfIterationsDone to 0.</summary>
		public void Reset()
		{
			this.Execute(0);
		}

		/// <summary>Set the observed value of the specified variable.</summary>
		/// <param name="variableName">Variable name</param>
		/// <param name="value">Observed value</param>
		public void SetObservedValue(string variableName, object value)
		{
			if (variableName=="winners") {
				this.winners = (int[])value;
				return ;
			}
			if (variableName=="losers") {
				this.losers = (int[])value;
				return ;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>
		/// Returns the marginal distribution for 'strengths' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public DistributionStructArray<Gaussian,double> StrengthsMarginal()
		{
			return this.strengths_marginal_F;
		}

		/// <summary>Update all marginals, by iterating message-passing an additional number of times</summary>
		/// <param name="additionalIterations">The number of iterations that should be executed, starting from the current message state.  Messages are not reset, even if observed values have changed.</param>
		public void Update(int additionalIterations)
		{
			this.Execute(checked(this.numberOfIterationsDone+additionalIterations), false);
		}

		/// <summary>
		/// Returns the marginal distribution for 'winners' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public PointMass<int[]> WinnersMarginal()
		{
			return this.winners_marginal_F;
		}

		#endregion

		#region Events
		/// <summary>Event that is fired when the progress of inference changes, typically at the end of one iteration of the inference algorithm.</summary>
		public event EventHandler<ProgressChangedEventArgs> ProgressChanged;
		#endregion

	}

}